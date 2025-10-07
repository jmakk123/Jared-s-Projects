"""Provides routes for v2"""

from utils.auth import require_api_key
from utils.helpers import get_data_by_year, get_price_data

BASE_URL = "/api/v2"


def register_v2_routes(app):
    """Register all routes for API version 2 using SQLite3."""

    @app.route(f"{BASE_URL}/<int:year>", methods=["GET"])
    @require_api_key
    def data_by_year(year):
        return get_data_by_year(year)

    @app.route(
        f"{BASE_URL}/<string:price_type>/<string:symbol>", methods=["GET"]
    )
    @require_api_key
    def price_data(price_type, symbol):
        return get_price_data(symbol, price_type)
