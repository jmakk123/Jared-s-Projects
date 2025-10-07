"""Provides routes for v4"""

from flask import jsonify, request

from utils.auth import require_api_key
from utils.helpers import (
    calculate_daily_return,
    evaluate_condition,
    fetch_trading_data,
    get_past_price,
    is_valid_trading_day,
    validate_request,
)
from utils.logger_decorators import (
    log_request_response,
    log_request_response_time,
)

BASE_URL = "/api/v4"

@log_request_response_time
@log_request_response
def back_test_route():
    """Handle the back-testing endpoint."""
    try:
        data = request.get_json()

        validate_request(data)

        start_date = data["start_date"]
        end_date = data["end_date"]

        if (
            not is_valid_trading_day(start_date) or
            not is_valid_trading_day(end_date)
        ):
            return jsonify(
                {"error": "Start or end date is not a valid trading day."}
                ), 400

        price_1_type, days_1 = data["value_1"][0], int(data["value_1"][1:])
        price_2_type, days_2 = data["value_2"][0], int(data["value_2"][1:])

        rows = fetch_trading_data(start_date, end_date)

        if not rows:
            return jsonify(
                {"error": "No data available for the given date range."}
                ), 400

        past_prices = {}

        total_return = 0.0
        num_observations = 0

        for row in rows:
            (
                date,
                symbol,
                open_price,
                high_price,
                low_price,
                close_price,
            ) = row[:6]

            if symbol not in past_prices:
                past_prices[symbol] = {}

            if price_1_type not in past_prices[symbol]:
                past_prices[symbol][price_1_type] = get_past_price(
                    symbol, date, price_1_type, days_1
                )

            price_1 = past_prices[symbol][price_1_type]

            if price_2_type not in past_prices[symbol]:
                past_prices[symbol][price_2_type] = get_past_price(
                    symbol, date, price_2_type, days_2
                )

            price_2 = past_prices[symbol][price_2_type]

            if price_1 is None or price_2 is None:
                continue

            condition_met = evaluate_condition(
                price_1, price_2, data["operator"]
            )

            if condition_met:
                num_observations += 1
                daily_return = calculate_daily_return(
                    {"Open": open_price, "Close": close_price},
                    data["purchase_type"],
                )
                total_return += daily_return

        return jsonify(
            {
                "return": round(total_return, 2),
                "num_observations": num_observations,
            }
        ), 200

    except ValueError as ve:
        return jsonify({"error": f"Error: {ve}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error: {str(e)}"}), 500

def register_v4_routes(app):
    """Register all routes for API version 4."""

    @app.route(f"{BASE_URL}/back_test", methods=["POST"])
    @require_api_key
    def handle_back_test_route():
        return back_test_route()
