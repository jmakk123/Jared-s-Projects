"""Provides the routes for v1.

Provides rows, unique stocks, and market counts.
"""

from flask import jsonify

from utils.auth import require_api_key
from utils.loading_utils import query_db
from utils.logger_decorators import (
    log_request_response,
    log_request_response_time,
)

BASE_URL = "/api/v1"


@log_request_response_time
@log_request_response
def row_count():
    """Calculate the number of rows in the stock data.

    Returns:
    - JSON: A JSON object containing the number of rows in the stock data.
    """
    query = "SELECT COUNT(*) FROM stocks"
    result = query_db(query, one=True)
    row_count = result[0] if result else 0
    return jsonify({"row_count": row_count})


@log_request_response_time
@log_request_response
def unique_stock_count():
    """Count the number of unique stocks in the stock data.

    Returns:
    - JSON: A JSON object containing the number of unique stocks
    in the stock data.
    """
    query = "SELECT COUNT(DISTINCT symbol) FROM stocks"
    result = query_db(query, one=True)
    unique_stock_count = result[0] if result else 0
    return jsonify({"unique_stock_count": unique_stock_count})


@log_request_response_time
@log_request_response
def row_by_market_count():
    """Count the number of rows in the stock data for each market.

    Returns:
    - JSON: A JSON object containing the number of rows in the stock
    data for each market.
    """
    nyse_query = "SELECT COUNT(*) FROM stocks WHERE market = 'NYSE'"
    nasdaq_query = "SELECT COUNT(*) FROM stocks WHERE market = 'NASDAQ'"

    nyse_result = query_db(nyse_query, one=True)
    nyse_count = nyse_result[0] if nyse_result else 0
    nasdaq_result = query_db(nasdaq_query, one=True)
    nasdaq_count = nasdaq_result[0] if nasdaq_result else 0

    return jsonify({"NYSE": nyse_count, "NASDAQ": nasdaq_count})


def register_v1_routes(app):
    """Register all routes for API."""

    @app.route(f"{BASE_URL}/row_count", methods=["GET"])
    @require_api_key
    def row_count_route():
        return row_count()

    @app.route(f"{BASE_URL}/unique_stock_count", methods=["GET"])
    @require_api_key
    def unique_stock_count_route():
        return unique_stock_count()

    @app.route(f"{BASE_URL}/row_by_market_count", methods=["GET"])
    @require_api_key
    def row_by_market_count_route():
        return row_by_market_count()
