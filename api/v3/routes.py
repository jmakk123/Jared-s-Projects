"""Provides routes for v3"""

from flask import jsonify, request

from utils.auth import require_api_key
from utils.loading_utils import execute_db, query_db
from utils.logger_decorators import (
    log_request_response,
    log_request_response_time,
)


@log_request_response_time
@log_request_response
def list_accounts():
    """List all accounts."""
    try:
        accounts = query_db("SELECT id AS account_id, name FROM accounts")
        return jsonify([dict(row) for row in accounts]), 200

    except Exception as e:
        return jsonify({"error": f"An error occured: {str(e)}"}), 500


@log_request_response_time
@log_request_response
def create_account():
    """Create a new account."""
    try:
        data = request.get_json()
        if not data or "name" not in data:
            return jsonify({"error": "Missing account name"}), 400

        existing_account = query_db(
            "SELECT * FROM accounts WHERE name = ?", [data["name"]], one=True
        )
        if existing_account:
            return jsonify({"error": "Account name already exists"}), 409

        execute_db("INSERT INTO accounts (name) VALUES (?)", (data["name"],))
        new_account = query_db(
            "SELECT id FROM accounts WHERE name = ?", [data["name"]], one=True
        )
        return jsonify({"account_id": new_account["id"]}), 201

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@log_request_response_time
@log_request_response
def delete_account():
    """Delete an account."""
    try:
        data = request.get_json()
        if not data or "account_id" not in data:
            return jsonify({"error": "Missing account_id"}), 400

        account = query_db(
            "SELECT * FROM accounts WHERE id = ?",
            [data["account_id"]],
            one=True,
        )
        if not account:
            return jsonify({"error": "Account not found"}), 404

        execute_db(
            "DELETE FROM stocks_owned WHERE account_id = ?",
            [data["account_id"]],
        )
        execute_db("DELETE FROM accounts WHERE id = ?", [data["account_id"]])
        return jsonify({"account_id": data["account_id"]}), 204

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@log_request_response_time
@log_request_response
def list_account_stocks(account_id):
    """List all stocks owned by an account."""
    try:
        account = query_db(
            "SELECT * FROM accounts WHERE id = ?", [account_id], one=True
        )
        if not account:
            return jsonify({"error": "Account not found"}), 404

        stocks = query_db(
            """
            SELECT symbol, purchase_date, sale_date, number_of_shares
            FROM stocks_owned
            WHERE account_id = ?
            """,
            [account_id],
        )
        holdings = [dict(row) for row in stocks]
        return jsonify(
            {
                "account_id": account_id,
                "name": account["name"],
                "stock_holdings": holdings,
            }
        ), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@log_request_response_time
@log_request_response
def stock_details(symbol):
    """List all holdings for a given stock."""
    try:
        stocks = query_db(
            """
            SELECT account_id, purchase_date, sale_date, number_of_shares
            FROM stocks_owned
            WHERE symbol = ?
            """,
            [symbol],
        )
        holdings = [dict(row) for row in stocks]
        return jsonify({"symbol": symbol, "holdings": holdings}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@log_request_response_time
@log_request_response
def add_stock():
    """Add a stock to an account."""
    try:
        data = request.get_json()
        required_fields = [
            "account_id",
            "symbol",
            "purchase_date",
            "sale_date",
            "number_of_shares",
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        query = """
                SELECT * FROM stocks
                WHERE Symbol = ? AND Date = ?
            """
        valid_entry = query_db(
            query, args=(data["symbol"], data["purchase_date"]), one=True
        )

        if not valid_entry:
            return jsonify({"error": "Invalid symbol or purchase date"}), 400

        insert_query = """
            INSERT INTO stocks_owned (
                account_id, symbol, purchase_date, sale_date,
                number_of_shares
            )
            VALUES (?, ?, ?, ?, ?)
        """
        execute_db(
            insert_query,
            args=[
                data["account_id"],
                data["symbol"],
                data["purchase_date"],
                data["sale_date"],
                data["number_of_shares"],
            ],
        )

        return "", 201

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@log_request_response_time
@log_request_response
def delete_stock():
    """Delete a stock holding."""
    try:
        data = request.get_json()
        required_fields = [
            "account_id",
            "symbol",
            "purchase_date",
            "sale_date",
            "number_of_shares",
        ]
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        args = [
            data["account_id"],
            data["symbol"],
            data["purchase_date"],
            data["sale_date"],
            data["number_of_shares"],
        ]

        query = """
                SELECT *
                FROM stocks_owned
                WHERE account_id = ? AND symbol = ? AND purchase_date = ?
                    AND sale_date = ? AND number_of_shares = ?
            """

        stock = query_db(query, args=args, one=True)

        if not stock:
            return jsonify({"error": "Stock holding not found"}), 404

        delete_query = """
                DELETE FROM stocks_owned
                WHERE account_id = ? AND symbol = ? AND purchase_date = ? 
                    AND sale_date = ? AND number_of_shares = ?
                """

        execute_db(delete_query, args=args)
        return "", 204

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@log_request_response_time
@log_request_response
def calculate_return(account_id):
    """Calculate nominal return for an account."""
    try:
        acct_query = "SELECT * FROM accounts WHERE id = ?"
        account = query_db(acct_query, args=(account_id,), one=True)
        if not account:
            return jsonify({"error": "Account not found"}), 404

        stocks_query = """
                SELECT symbol, purchase_date, sale_date, number_of_shares
                FROM stocks_owned
                WHERE account_id = ?
                """

        stocks = query_db(stocks_query, args=(account_id,))
        total_return = 0.0

        for stock in stocks:
            symbol, purchase_date, sale_date, number_of_shares = stock

            sale_query = """
            SELECT Close FROM stocks WHERE Symbol = ? AND Date = ?
            """
            purchase_query = """
            SELECT Open FROM stocks WHERE Symbol = ? AND Date = ?
            """

            sale_data = query_db(
                sale_query, args=(symbol, sale_date), one=True
            )
            purchase_data = query_db(
                purchase_query, args=(symbol, purchase_date), one=True
            )

            if sale_data and purchase_data:
                total_return += number_of_shares * (
                    sale_data[0] - purchase_data[0]
                )

        return jsonify({"account_id": account_id, "return": total_return}), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


def register_v3_routes(app):
    """Register all routes for API version 3."""
    BASE_URL = "/api/v3"

    @app.route(f"{BASE_URL}/accounts", methods=["GET", "POST", "DELETE"])
    @require_api_key
    def base_url_handler():
        if request.method == "GET":
            return list_accounts()
        if request.method == "POST":
            return create_account()
        if request.method == "DELETE":
            return delete_account()

    @app.route(f"{BASE_URL}/accounts/<int:account_id>", methods=["GET"])
    @require_api_key
    def stock_lister(account_id):
        return list_account_stocks(account_id)

    @app.route(f"{BASE_URL}/stocks/<string:symbol>", methods=["GET"])
    @require_api_key
    def stock_details_route(symbol):
        return stock_details(symbol)

    @app.route(f"{BASE_URL}/stocks", methods=["POST", "DELETE"])
    @require_api_key
    def stock_url_handler():
        if request.method == "POST":
            return add_stock()
        if request.method == "DELETE":
            return delete_stock()

    @app.route(f"{BASE_URL}/accounts/return/<int:account_id>", methods=["GET"])
    @require_api_key
    def return_calculator(account_id):
        return calculate_return(account_id)
