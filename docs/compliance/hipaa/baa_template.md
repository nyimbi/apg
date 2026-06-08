# HIPAA Business Associate Agreement (BAA) Template

**IMPORTANT**: This template must be reviewed and executed by qualified legal counsel
before use. It is provided for reference only and does not constitute legal advice.

---

## BUSINESS ASSOCIATE AGREEMENT

This Business Associate Agreement ("BAA") is entered into as of **[DATE]** ("Effective Date")
between:

**Covered Entity**: [Customer Organization Name], a [entity type] organized under the laws
of [jurisdiction], with its principal place of business at [address] ("Covered Entity")

**Business Associate**: Datacraft Ltd., a limited liability company organized under the
laws of Kenya, with its principal place of business at Nairobi, Kenya ("Business Associate")

### Recitals

WHEREAS, Business Associate provides APG Platform services (software as a service) to
Covered Entity under a separate Service Agreement ("Underlying Agreement");

WHEREAS, in providing such services, Business Associate may create, receive, maintain,
or transmit Protected Health Information ("PHI") on behalf of Covered Entity;

WHEREAS, the parties desire to enter into this BAA to comply with 45 CFR Parts 160 and
164 (the "HIPAA Rules");

---

### 1. Definitions

**"PHI"** means Protected Health Information as defined in 45 CFR § 160.103, limited to
information that Business Associate creates, receives, maintains, or transmits on behalf
of Covered Entity.

**"Electronic PHI"** or **"ePHI"** means PHI that is transmitted by or maintained in
electronic media as defined in 45 CFR § 160.103.

**"Breach"** means the acquisition, access, use, or disclosure of PHI in a manner not
permitted under 45 CFR Part 164, Subpart E.

**"Security Incident"** means the attempted or successful unauthorized access, use,
disclosure, modification, or destruction of information or interference with system
operations in an information system.

---

### 2. Business Associate Obligations

#### 2.1 Use and Disclosure Restrictions

Business Associate agrees to:

(a) Use or disclose PHI only as permitted or required by this BAA or as required by law;

(b) Not use or disclose PHI in a manner that would violate Subpart E of 45 CFR Part 164
if done by Covered Entity;

(c) Use appropriate safeguards, and comply with Subpart C of 45 CFR Part 164 with respect
to ePHI, to prevent use or disclosure of PHI other than as provided for by this BAA;

(d) Report to Covered Entity any use or disclosure of PHI not provided for by this BAA
of which Business Associate becomes aware, including Breaches of Unsecured PHI as required
by 45 CFR § 164.410, and any Security Incidents of which it becomes aware.

#### 2.2 Technical Safeguards (APG Platform)

Business Associate implements the following technical safeguards:

| Safeguard | Implementation | HIPAA Reference |
|-----------|---------------|----------------|
| Access control | OPA policy-as-code (policies/apg/capabilities/healthcare.rego) | §164.312(a) |
| Audit controls | Immutable audit log (apg_audit_events + NATS events) | §164.312(b) |
| Integrity | SHA-256 tamper-evident hash chain in audit table | §164.312(c) |
| Transmission security | TLS 1.3 minimum for all API endpoints | §164.312(e) |
| PHI field classification | capabilities/common/phi/classifier.py | §164.514(b) |
| Minimum necessary | OPA minimum_necessary enforcement per purpose | §164.502(b) |

#### 2.3 Subcontractors

Business Associate shall ensure that any subcontractor that creates, receives, maintains,
or transmits PHI on behalf of Business Associate agrees to the same restrictions and
conditions as apply to Business Associate under this BAA.

#### 2.4 Access

Within **ten (10) business days** of Covered Entity's written request, Business Associate
shall make available PHI in a Designated Record Set to Covered Entity as necessary for
Covered Entity to comply with 45 CFR § 164.524.

#### 2.5 Breach Notification

Business Associate shall notify Covered Entity of a Breach of Unsecured PHI within
**thirty (30) calendar days** of discovery. Notification shall include:

(a) Identification of each individual whose Unsecured PHI has been or is reasonably
believed to have been accessed, acquired, used, or disclosed;

(b) A brief description of what happened, including the date of the Breach and date
of discovery;

(c) A description of the types of Unsecured PHI involved;

(d) Steps individuals should take to protect themselves from potential harm;

(e) A brief description of what Business Associate is doing to investigate, mitigate
harm, and protect against further Breaches.

---

### 3. Covered Entity Obligations

Covered Entity agrees to:

(a) Notify Business Associate of any limitation(s) in the notice of privacy practices
that may affect Business Associate's use or disclosure of PHI;

(b) Notify Business Associate of any changes in, or revocation of, permission by
an Individual to use or disclose PHI;

(c) Not request Business Associate to use or disclose PHI in any manner that would
not be permissible under Subpart E of 45 CFR Part 164 if done by Covered Entity.

---

### 4. Term and Termination

#### 4.1 Term

This BAA shall be effective as of the Effective Date and shall terminate when all PHI
received from or created by Business Associate on behalf of Covered Entity is destroyed
or returned to Covered Entity, or, if return or destruction is infeasible, protections
are extended.

#### 4.2 Termination for Cause

Covered Entity may terminate this BAA and the Underlying Agreement if Business Associate
has violated a material term of this BAA and Business Associate fails to cure within
**thirty (30) days** of receipt of written notice.

#### 4.3 Effect of Termination

Upon termination for any reason, Business Associate shall, as directed by Covered Entity:
return or destroy all PHI received from or created or received by Business Associate on
behalf of Covered Entity. If return or destruction is infeasible, Business Associate shall
retain only that PHI that cannot feasibly be returned or destroyed and shall extend the
protections of this BAA to such PHI.

---

### 5. Miscellaneous

#### 5.1 Regulatory References

Any reference in this BAA to a section of the HIPAA Rules shall mean the section as
in effect or as amended, including the HITECH Act requirements incorporated therein.

#### 5.2 Interpretation

Any ambiguity in this BAA shall be resolved in favor of a meaning that permits Covered
Entity to comply with the HIPAA Rules.

#### 5.3 No Third-Party Beneficiaries

Nothing in this BAA shall confer any rights or remedies upon any person other than the
parties and their respective successors and permitted assigns.

---

### Signatures

**COVERED ENTITY**

By: ________________________  
Name: ______________________  
Title: _____________________  
Date: ______________________

**BUSINESS ASSOCIATE (Datacraft Ltd.)**

By: ________________________  
Name: ______________________  
Title: _____________________  
Date: ______________________

---

*This template was prepared in reference to 45 CFR §§ 164.502(e), 164.504(e).
Have this document reviewed by qualified healthcare legal counsel before execution.*
