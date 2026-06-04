-- =============================================================================
-- APG Payroll — PostgreSQL normalized schema
-- Run: psql $DATABASE_URL < database/schema.sql
-- © 2025 Datacraft
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ---------------------------------------------------------------------------
-- Helpers
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
	NEW.updated_at = now();
	RETURN NEW;
END;
$$;

-- =============================================================================
-- 1. Payroll Configuration (per tenant/country)
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_payroll_config (
	id                              TEXT        PRIMARY KEY,
	tenant_id                       TEXT        NOT NULL,
	country                         TEXT        NOT NULL,
	currency                        TEXT        NOT NULL DEFAULT 'KES',
	default_pay_frequency           TEXT        NOT NULL DEFAULT 'monthly',
	employer_name                   TEXT        NOT NULL,
	employer_tax_pin                TEXT        NOT NULL,
	employer_nssf_code              TEXT,
	employer_nhif_code              TEXT,
	pension_scheme_type             TEXT        NOT NULL DEFAULT 'defined_contribution',
	pension_employee_rate           NUMERIC(7,4) NOT NULL DEFAULT 0.06,
	pension_employer_rate           NUMERIC(7,4) NOT NULL DEFAULT 0.06,
	transport_allowance_taxfree_limit NUMERIC(15,2) NOT NULL DEFAULT 2000.00,
	overtime_multiplier_standard    NUMERIC(5,2) NOT NULL DEFAULT 1.5,
	overtime_multiplier_holiday     NUMERIC(5,2) NOT NULL DEFAULT 2.0,
	gl_salary_account               TEXT        NOT NULL DEFAULT '5100',
	gl_paye_liability_account       TEXT        NOT NULL DEFAULT '2210',
	gl_nssf_liability_account       TEXT        NOT NULL DEFAULT '2220',
	gl_nhif_liability_account       TEXT        NOT NULL DEFAULT '2230',
	created_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                      TEXT        NOT NULL DEFAULT 'system',
	is_deleted                      BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_config_tenant ON pr_payroll_config (tenant_id) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_config_updated_at BEFORE UPDATE ON pr_payroll_config
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 2. Employees (payroll view — lightweight snapshot)
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_employees (
	id                      TEXT        PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	employee_number         TEXT        NOT NULL,
	full_name               TEXT        NOT NULL,
	national_id             TEXT        NOT NULL,
	tax_pin                 TEXT        NOT NULL,
	nssf_number             TEXT,
	nhif_number             TEXT,
	bank_code               TEXT,
	bank_branch_code        TEXT,
	bank_account_number     TEXT,
	bank_account_name       TEXT,
	mobile_money_number     TEXT,
	department_id           TEXT,
	department_name         TEXT,
	cost_center             TEXT,
	employment_type         TEXT        NOT NULL DEFAULT 'permanent',
	hire_date               DATE        NOT NULL,
	termination_date        DATE,
	salary_grade            TEXT,
	basic_salary            NUMERIC(15,2) NOT NULL DEFAULT 0,
	currency                TEXT        NOT NULL DEFAULT 'KES',
	country                 TEXT        NOT NULL DEFAULT 'KE',
	payment_method          TEXT        NOT NULL DEFAULT 'bank_eft',
	pay_frequency           TEXT        NOT NULL DEFAULT 'monthly',
	is_expatriate           BOOLEAN     NOT NULL DEFAULT FALSE,
	tax_exemption_certificate TEXT,
	is_active               BOOLEAN     NOT NULL DEFAULT TRUE,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, employee_number)
);
CREATE INDEX IF NOT EXISTS idx_pr_emp_tenant ON pr_employees (tenant_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pr_emp_number ON pr_employees (tenant_id, employee_number);
CREATE INDEX IF NOT EXISTS idx_pr_emp_name_trgm ON pr_employees USING gin (full_name gin_trgm_ops);
CREATE TRIGGER trg_pr_emp_updated_at BEFORE UPDATE ON pr_employees
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 3. Pay Periods
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_pay_periods (
	id              TEXT        PRIMARY KEY,
	tenant_id       TEXT        NOT NULL,
	period_code     TEXT        NOT NULL,
	pay_frequency   TEXT        NOT NULL,
	start_date      DATE        NOT NULL,
	end_date        DATE        NOT NULL,
	pay_date        DATE        NOT NULL,
	status          TEXT        NOT NULL DEFAULT 'open',
	currency        TEXT        NOT NULL DEFAULT 'KES',
	country         TEXT        NOT NULL DEFAULT 'KE',
	notes           TEXT,
	created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by      TEXT        NOT NULL DEFAULT 'system',
	is_deleted      BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, period_code)
);
CREATE INDEX IF NOT EXISTS idx_pr_period_tenant ON pr_pay_periods (tenant_id, status) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_period_updated_at BEFORE UPDATE ON pr_pay_periods
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 4. Payroll Runs
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_payroll_runs (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	period_id           TEXT        NOT NULL REFERENCES pr_pay_periods(id),
	run_number          INTEGER     NOT NULL DEFAULT 1,
	status              TEXT        NOT NULL DEFAULT 'draft',
	description         TEXT,
	is_bonus_run        BOOLEAN     NOT NULL DEFAULT FALSE,
	is_supplementary    BOOLEAN     NOT NULL DEFAULT FALSE,
	total_gross         NUMERIC(15,2) NOT NULL DEFAULT 0,
	total_deductions    NUMERIC(15,2) NOT NULL DEFAULT 0,
	total_taxes         NUMERIC(15,2) NOT NULL DEFAULT 0,
	total_net           NUMERIC(15,2) NOT NULL DEFAULT 0,
	employee_count      INTEGER     NOT NULL DEFAULT 0,
	approved_by         TEXT,
	approved_at         TIMESTAMPTZ,
	posted_by           TEXT,
	posted_at           TIMESTAMPTZ,
	reversed_by         TEXT,
	reversal_reason     TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_run_tenant ON pr_payroll_runs (tenant_id, status) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pr_run_period ON pr_payroll_runs (period_id);
CREATE TRIGGER trg_pr_run_updated_at BEFORE UPDATE ON pr_payroll_runs
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 5. Payslip Lines (all earnings and deductions per employee per run)
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_payslip_lines (
	id                      TEXT        PRIMARY KEY,
	tenant_id               TEXT        NOT NULL,
	run_id                  TEXT        NOT NULL REFERENCES pr_payroll_runs(id),
	employee_id             TEXT        NOT NULL REFERENCES pr_employees(id),
	element_type            TEXT        NOT NULL,
	element_name            TEXT        NOT NULL,
	amount                  NUMERIC(15,2) NOT NULL DEFAULT 0,
	is_taxable              BOOLEAN     NOT NULL DEFAULT TRUE,
	is_pensionable          BOOLEAN     NOT NULL DEFAULT TRUE,
	is_employer_contribution BOOLEAN    NOT NULL DEFAULT FALSE,
	notes                   TEXT,
	created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by              TEXT        NOT NULL DEFAULT 'system',
	is_deleted              BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_line_run ON pr_payslip_lines (run_id, employee_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pr_line_emp ON pr_payslip_lines (employee_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pr_line_type ON pr_payslip_lines (run_id, element_type) WHERE NOT is_deleted;
-- Partition hint: for large datasets partition by run_id or date range
CREATE TRIGGER trg_pr_line_updated_at BEFORE UPDATE ON pr_payslip_lines
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 6. PAYE Tax Calculations
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_tax_calculations (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	run_id              TEXT        NOT NULL REFERENCES pr_payroll_runs(id),
	employee_id         TEXT        NOT NULL REFERENCES pr_employees(id),
	country             TEXT        NOT NULL,
	gross_income        NUMERIC(15,2) NOT NULL,
	taxable_income      NUMERIC(15,2) NOT NULL,
	personal_relief     NUMERIC(15,2) NOT NULL DEFAULT 0,
	insurance_relief    NUMERIC(15,2) NOT NULL DEFAULT 0,
	mortgage_relief     NUMERIC(15,2) NOT NULL DEFAULT 0,
	other_relief        NUMERIC(15,2) NOT NULL DEFAULT 0,
	bands_applied       JSONB        NOT NULL DEFAULT '[]',
	gross_tax           NUMERIC(15,2) NOT NULL DEFAULT 0,
	tax_relief_total    NUMERIC(15,2) NOT NULL DEFAULT 0,
	paye_amount         NUMERIC(15,2) NOT NULL DEFAULT 0,
	tax_code            TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (run_id, employee_id)
);
CREATE INDEX IF NOT EXISTS idx_pr_tax_run ON pr_tax_calculations (run_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pr_tax_emp ON pr_tax_calculations (employee_id) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_tax_updated_at BEFORE UPDATE ON pr_tax_calculations
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 7. Statutory Deductions
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_statutory_deductions (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	run_id              TEXT        NOT NULL REFERENCES pr_payroll_runs(id),
	employee_id         TEXT        NOT NULL REFERENCES pr_employees(id),
	country             TEXT        NOT NULL,
	deduction_type      TEXT        NOT NULL,
	employee_amount     NUMERIC(15,2) NOT NULL DEFAULT 0,
	employer_amount     NUMERIC(15,2) NOT NULL DEFAULT 0,
	basis               NUMERIC(15,2) NOT NULL DEFAULT 0,
	rate_used           NUMERIC(7,4),
	cap_applied         BOOLEAN     NOT NULL DEFAULT FALSE,
	notes               TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_stat_run ON pr_statutory_deductions (run_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pr_stat_emp ON pr_statutory_deductions (employee_id) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_stat_updated_at BEFORE UPDATE ON pr_statutory_deductions
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 8. Leave Balances
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_leave_balances (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	employee_id         TEXT        NOT NULL REFERENCES pr_employees(id),
	leave_type          TEXT        NOT NULL,
	year                INTEGER     NOT NULL,
	entitled_days       NUMERIC(7,2) NOT NULL DEFAULT 0,
	taken_days          NUMERIC(7,2) NOT NULL DEFAULT 0,
	carried_forward     NUMERIC(7,2) NOT NULL DEFAULT 0,
	encashed_days       NUMERIC(7,2) NOT NULL DEFAULT 0,
	encashed_amount     NUMERIC(15,2) NOT NULL DEFAULT 0,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, employee_id, leave_type, year)
);
CREATE INDEX IF NOT EXISTS idx_pr_leave_emp ON pr_leave_balances (employee_id, year) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_leave_updated_at BEFORE UPDATE ON pr_leave_balances
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 9. GL Journal Entries
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_gl_entries (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	run_id              TEXT        NOT NULL REFERENCES pr_payroll_runs(id),
	journal_date        DATE        NOT NULL,
	account_code        TEXT        NOT NULL,
	account_name        TEXT        NOT NULL,
	entry_type          TEXT        NOT NULL CHECK (entry_type IN ('debit','credit')),
	amount              NUMERIC(15,2) NOT NULL,
	cost_center         TEXT,
	department_code     TEXT,
	reference           TEXT,
	narration           TEXT,
	is_posted           BOOLEAN     NOT NULL DEFAULT FALSE,
	posted_at           TIMESTAMPTZ,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_gl_run ON pr_gl_entries (run_id) WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_pr_gl_account ON pr_gl_entries (tenant_id, account_code) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_gl_updated_at BEFORE UPDATE ON pr_gl_entries
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 10. Salary Advances
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_advances (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	employee_id                 TEXT        NOT NULL REFERENCES pr_employees(id),
	amount                      NUMERIC(15,2) NOT NULL,
	disbursement_date           DATE        NOT NULL,
	recovery_start_period_id    TEXT,
	monthly_recovery            NUMERIC(15,2) NOT NULL,
	amount_recovered            NUMERIC(15,2) NOT NULL DEFAULT 0,
	approved_by                 TEXT        NOT NULL,
	notes                       TEXT,
	is_fully_recovered          BOOLEAN     NOT NULL DEFAULT FALSE,
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL DEFAULT 'system',
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_advance_emp ON pr_advances (employee_id) WHERE NOT is_deleted AND NOT is_fully_recovered;
CREATE TRIGGER trg_pr_advance_updated_at BEFORE UPDATE ON pr_advances
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 11. Garnishments / Court Orders
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_garnishments (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	employee_id         TEXT        NOT NULL REFERENCES pr_employees(id),
	court_reference     TEXT        NOT NULL,
	creditor_name       TEXT        NOT NULL,
	monthly_amount      NUMERIC(15,2) NOT NULL,
	total_order_amount  NUMERIC(15,2),
	amount_deducted     NUMERIC(15,2) NOT NULL DEFAULT 0,
	effective_date      DATE        NOT NULL,
	expiry_date         DATE,
	is_active           BOOLEAN     NOT NULL DEFAULT TRUE,
	notes               TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_garnish_emp ON pr_garnishments (employee_id) WHERE NOT is_deleted AND is_active;
CREATE TRIGGER trg_pr_garnish_updated_at BEFORE UPDATE ON pr_garnishments
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 12. Overtime Records
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_overtime (
	id                  TEXT        PRIMARY KEY,
	tenant_id           TEXT        NOT NULL,
	employee_id         TEXT        NOT NULL REFERENCES pr_employees(id),
	run_id              TEXT        NOT NULL REFERENCES pr_payroll_runs(id),
	hours               NUMERIC(7,2) NOT NULL,
	rate_multiplier     NUMERIC(5,2) NOT NULL DEFAULT 1.5,
	hourly_rate         NUMERIC(15,2) NOT NULL DEFAULT 0,
	computed_amount     NUMERIC(15,2) NOT NULL DEFAULT 0,
	approved_by         TEXT,
	created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by          TEXT        NOT NULL DEFAULT 'system',
	is_deleted          BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_pr_ot_run ON pr_overtime (run_id) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_ot_updated_at BEFORE UPDATE ON pr_overtime
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 13. Final Settlement / Terminal Benefits
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_final_settlements (
	id                          TEXT        PRIMARY KEY,
	tenant_id                   TEXT        NOT NULL,
	employee_id                 TEXT        NOT NULL REFERENCES pr_employees(id),
	termination_date            DATE        NOT NULL,
	last_day_worked             DATE        NOT NULL,
	reason_for_leaving          TEXT        NOT NULL,
	notice_period_days          INTEGER     NOT NULL DEFAULT 0,
	notice_period_served_days   INTEGER     NOT NULL DEFAULT 0,
	prorated_salary             NUMERIC(15,2) NOT NULL DEFAULT 0,
	leave_encashment            NUMERIC(15,2) NOT NULL DEFAULT 0,
	notice_pay                  NUMERIC(15,2) NOT NULL DEFAULT 0,
	severance_pay               NUMERIC(15,2) NOT NULL DEFAULT 0,
	gratuity                    NUMERIC(15,2) NOT NULL DEFAULT 0,
	other_benefits              NUMERIC(15,2) NOT NULL DEFAULT 0,
	total_gross                 NUMERIC(15,2) NOT NULL DEFAULT 0,
	paye_on_settlement          NUMERIC(15,2) NOT NULL DEFAULT 0,
	net_settlement              NUMERIC(15,2) NOT NULL DEFAULT 0,
	benefit_lines               JSONB        NOT NULL DEFAULT '[]',
	run_id                      TEXT,
	status                      TEXT        NOT NULL DEFAULT 'draft',
	created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
	created_by                  TEXT        NOT NULL DEFAULT 'system',
	is_deleted                  BOOLEAN     NOT NULL DEFAULT FALSE,
	UNIQUE (tenant_id, employee_id)
);
CREATE INDEX IF NOT EXISTS idx_pr_settlement_emp ON pr_final_settlements (employee_id) WHERE NOT is_deleted;
CREATE TRIGGER trg_pr_settlement_updated_at BEFORE UPDATE ON pr_final_settlements
	FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- =============================================================================
-- 14. Domain Events (audit log)
-- =============================================================================
CREATE TABLE IF NOT EXISTS pr_domain_events (
	id              BIGSERIAL   PRIMARY KEY,
	tenant_id       TEXT        NOT NULL,
	event_type      TEXT        NOT NULL,
	aggregate_id    TEXT        NOT NULL,
	aggregate_type  TEXT        NOT NULL,
	payload         JSONB       NOT NULL DEFAULT '{}',
	actor_id        TEXT,
	occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_pr_events_tenant ON pr_domain_events (tenant_id, event_type);
CREATE INDEX IF NOT EXISTS idx_pr_events_aggregate ON pr_domain_events (aggregate_id);
-- Partition by month for high-volume tenants:
-- PARTITION BY RANGE (occurred_at)

-- =============================================================================
-- Views
-- =============================================================================

-- Payroll run summary
CREATE OR REPLACE VIEW v_pr_run_summary AS
SELECT
	r.id AS run_id,
	r.tenant_id,
	p.period_code,
	p.pay_date,
	r.status,
	r.employee_count,
	r.total_gross,
	r.total_deductions,
	r.total_taxes,
	r.total_net,
	r.approved_by,
	r.approved_at,
	r.posted_at,
	r.created_at
FROM pr_payroll_runs r
JOIN pr_pay_periods p ON p.id = r.period_id
WHERE NOT r.is_deleted;

-- Employee payslip totals per run
CREATE OR REPLACE VIEW v_pr_employee_payslip AS
SELECT
	l.run_id,
	l.employee_id,
	l.tenant_id,
	e.employee_number,
	e.full_name,
	SUM(l.amount) FILTER (WHERE l.element_type IN ('basic','allowance','overtime','bonus','commission','back_pay')) AS gross_pay,
	SUM(l.amount) FILTER (WHERE l.element_type IN ('paye')) AS paye,
	SUM(l.amount) FILTER (WHERE l.element_type IN ('nssf','nhif','nhif_shi','nita','sdl','wcf','napsa','ssnit','pencom','pension')) AS statutory_deductions,
	SUM(l.amount) FILTER (WHERE l.element_type IN ('loan_recovery','garnishment','advance_recovery','other_deduction')) AS other_deductions,
	SUM(l.amount) FILTER (WHERE NOT l.is_employer_contribution) AS total_employee_amount
FROM pr_payslip_lines l
JOIN pr_employees e ON e.id = l.employee_id
WHERE NOT l.is_deleted
GROUP BY l.run_id, l.employee_id, l.tenant_id, e.employee_number, e.full_name;
