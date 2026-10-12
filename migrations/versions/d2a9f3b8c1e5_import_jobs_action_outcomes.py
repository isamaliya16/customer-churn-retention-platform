"""Add import_jobs, retention action outcome/due_date/offer_cost, and a latest-prediction index

Revision ID: d2a9f3b8c1e5
Revises: c7d41e9a2b60
Create Date: 2026-10-03 19:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = 'd2a9f3b8c1e5'
down_revision = 'c7d41e9a2b60'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('import_jobs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('filename', sa.String(length=255), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('phase', sa.String(length=24), nullable=False),
    sa.Column('mode', sa.String(length=8), nullable=False),
    sa.Column('score_after', sa.Boolean(), nullable=False),
    sa.Column('total_rows', sa.Integer(), nullable=False),
    sa.Column('processed_rows', sa.Integer(), nullable=False),
    sa.Column('created_count', sa.Integer(), nullable=False),
    sa.Column('updated_count', sa.Integer(), nullable=False),
    sa.Column('skipped_count', sa.Integer(), nullable=False),
    sa.Column('rejected_count', sa.Integer(), nullable=False),
    sa.Column('scored_count', sa.Integer(), nullable=False),
    sa.Column('errors', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('warnings', sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql'), nullable=False),
    sa.Column('message', sa.String(length=500), nullable=True),
    sa.Column('created_by', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('finished_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('import_jobs', schema=None) as batch_op:
        batch_op.create_index('ix_import_jobs_created_at', ['created_at'], unique=False)

    with op.batch_alter_table('retention_actions', schema=None) as batch_op:
        batch_op.add_column(sa.Column('due_date', sa.Date(), nullable=True))
        batch_op.add_column(sa.Column('offer_cost', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('outcome', sa.String(length=16), nullable=True))
        batch_op.create_index('ix_retention_actions_outcome', ['outcome'], unique=False)

    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.create_index('ix_predictions_customer_id_id', ['customer_id', 'id'], unique=False)


def downgrade():
    with op.batch_alter_table('predictions', schema=None) as batch_op:
        batch_op.drop_index('ix_predictions_customer_id_id')

    with op.batch_alter_table('retention_actions', schema=None) as batch_op:
        batch_op.drop_index('ix_retention_actions_outcome')
        batch_op.drop_column('outcome')
        batch_op.drop_column('offer_cost')
        batch_op.drop_column('due_date')

    with op.batch_alter_table('import_jobs', schema=None) as batch_op:
        batch_op.drop_index('ix_import_jobs_created_at')
    op.drop_table('import_jobs')
