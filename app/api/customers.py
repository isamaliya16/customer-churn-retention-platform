"""Customer CRUD (+ billing transactions).

RBAC: admin and analyst can read/create/update; only admin can delete.
"""

from flask.views import MethodView
from flask_smorest import Blueprint

from app.api.helpers import errors
from app.schemas.common_schema import page_envelope, paginated
from app.schemas.customer_schema import (
    CustomerQuerySchema,
    CustomerSchema,
    CustomerUpdateSchema,
    TransactionSchema,
)
from app.services.customer_service import CustomerService
from app.utils.cache import invalidate_analytics_cache
from app.utils.decorators import role_required

blp = Blueprint("customers", __name__, url_prefix="/customers", description="Customer CRUD and billing history")


@blp.route("")
class Customers(MethodView):
    @errors(blp, 401, 403)
    @role_required("admin", "analyst")
    @blp.arguments(CustomerQuerySchema, location="query")
    @blp.response(200, paginated(CustomerSchema))
    def get(self, args):
        """List customers: search (q), filter (risk_level / contract_type / is_active), sort, paginate."""
        page, per_page = args.pop("page"), args.pop("per_page")
        items, total = CustomerService().list_customers(page, per_page, **args)
        return page_envelope(items, total, page, per_page)

    @errors(blp, 401, 403, 409)
    @role_required("admin", "analyst")
    @blp.arguments(CustomerSchema)
    @blp.response(201, CustomerSchema)
    def post(self, payload):
        """Create a customer. Only name, email, monthly_charges and contract_type are required."""
        customer = CustomerService().create(payload)
        invalidate_analytics_cache()
        return CustomerService().get_with_risk(customer.id)


@blp.route("/<int:customer_id>")
class CustomerDetail(MethodView):
    @errors(blp, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.response(200, CustomerSchema)
    def get(self, customer_id):
        """Fetch a customer (with its latest risk level)."""
        return CustomerService().get_with_risk(customer_id)

    @errors(blp, 401, 403, 404, 409)
    @role_required("admin", "analyst")
    @blp.arguments(CustomerUpdateSchema)
    @blp.response(200, CustomerSchema)
    def put(self, payload, customer_id):
        """Update a customer (partial update). Setting is_active=false records the churn date."""
        customer = CustomerService().update(customer_id, payload)
        invalidate_analytics_cache()
        return customer

    @errors(blp, 401, 403, 404)
    @role_required("admin")
    @blp.response(204)
    def delete(self, customer_id):
        """Delete a customer and its predictions/transactions/actions. Admin only."""
        CustomerService().delete(customer_id)
        invalidate_analytics_cache()
        return None


@blp.route("/<int:customer_id>/transactions")
class CustomerTransactions(MethodView):
    @errors(blp, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.response(200, TransactionSchema(many=True))
    def get(self, customer_id):
        """Most recent billing transactions for a customer (max 50)."""
        return CustomerService().list_transactions(customer_id)

    @errors(blp, 401, 403, 404)
    @role_required("admin", "analyst")
    @blp.arguments(TransactionSchema)
    @blp.response(201, TransactionSchema)
    def post(self, payload, customer_id):
        """Record a billing transaction."""
        return CustomerService().add_transaction(customer_id, payload)
