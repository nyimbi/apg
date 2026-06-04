"""Initial migration: create apg_records JSONB store.

Revision ID: 0001
Revises: 
Create Date: 2026-06-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'apg_records',
        sa.Column('id', sa.Text(), nullable=False),
        sa.Column('collection', sa.Text(), nullable=False),
        sa.Column('tenant_id', sa.Text(), nullable=False, server_default='default'),
        sa.Column('data', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('collection', 'id'),
    )
    op.create_index('idx_apg_records_tenant', 'apg_records', ['collection', 'tenant_id'])
    op.create_index('idx_apg_records_data', 'apg_records', ['data'],
                    postgresql_using='gin', postgresql_ops={'data': 'jsonb_ops'})


def downgrade() -> None:
    op.drop_index('idx_apg_records_data', 'apg_records')
    op.drop_index('idx_apg_records_tenant', 'apg_records')
    op.drop_table('apg_records')
