"""Add revoked_tokens table (JWT logout / revocation)

Revision ID: c7d41e9a2b60
Revises: 4b504505fa50
Create Date: 2026-10-02 08:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c7d41e9a2b60'
down_revision = '4b504505fa50'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('revoked_tokens',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('jti', sa.String(length=64), nullable=False),
    sa.Column('token_type', sa.String(length=10), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('expires_at', sa.DateTime(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('revoked_tokens', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_revoked_tokens_jti'), ['jti'], unique=True)
        batch_op.create_index('ix_revoked_tokens_expires_at', ['expires_at'], unique=False)


def downgrade():
    with op.batch_alter_table('revoked_tokens', schema=None) as batch_op:
        batch_op.drop_index('ix_revoked_tokens_expires_at')
        batch_op.drop_index(batch_op.f('ix_revoked_tokens_jti'))
    op.drop_table('revoked_tokens')
