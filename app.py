import dash
import dash_bootstrap_components as dbc
from dash import html
from dash.dependencies import Input, Output

from src.agent import TrainAgentInterface
from src.othello import Board, Othello
from src.utils.dataclass import Pass, Place
from src.utils.enums import Stone
from src.utils.util import Turn, change_stone

# Load your model
agent = TrainAgentInterface.load("output/agent.pkl")
agent.test()

# Create the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize the game
SIZE = 6
user_stone = Stone.BLACK
ai_stone = Stone.WHITE
game = Othello(size=SIZE)


def create_board(board: Board) -> html.Table:
    table_style = {
        "borderCollapse": "collapse",
    }
    cell_style = {
        "width": "40px",
        "height": "40px",
        "border": "1px solid black",
        "padding": "0",
        "backgroundColor": "#a0d9a0",  # Common board color
    }
    base_stone_style = {
        "width": "80%",
        "height": "80%",
        "borderRadius": "16px",  # 40px * 0.8 / 2
        "margin": "auto",
        "padding": "0",
        "border": "none",
        "display": "block",
    }
    black_stone_style = base_stone_style | {
        "backgroundColor": "black",
    }
    white_stone_style = base_stone_style | {
        "backgroundColor": "white",
    }
    empty_stone_style = {
        "width": "100%",
        "height": "100%",
        "opacity": 0,
    }

    return html.Table(
        style=table_style,
        children=[
            html.Tr(
                [
                    html.Td(
                        style=cell_style,
                        children=[
                            html.Button(
                                "",
                                style=(
                                    black_stone_style
                                    if board[Place(y=i, x=j)] == Stone.BLACK
                                    else white_stone_style
                                    if board[Place(y=i, x=j)] == Stone.WHITE
                                    else empty_stone_style
                                ),
                                id="button-{}-{}".format(i, j),
                                disabled=board[Place(y=i, x=j)] != Stone.EMPTY,
                            )
                        ],
                    )
                    for j in range(SIZE)
                ]
            )
            for i in range(SIZE)
        ],
    )


def layout() -> dbc.Container:
    return dbc.Container(
        [
            html.Div(
                create_board(game.get_board()),
                id="board-container",
                className="d-flex align-items-center justify-content-center",
            ),
            html.Div(
                className="my-1",
                children=[
                    html.Button(
                        "Reset",
                        id="reset-button",
                        n_clicks=0,
                        className="mx-auto",
                    ),
                    html.Button(
                        "Pass",
                        id="pass-button",
                        n_clicks=0,
                        className="mx-auto",
                    ),
                ],
            ),
            html.Div(
                f'You are playing as {"Black" if user_stone == Stone.BLACK else "White"}',
                id="user-stone",
                className="mx-auto",
            ),
            html.P("", id="message", className="mx-auto"),
        ],
        style={"height": "100vh"},
        className="m-5 text-center",
    )


app.layout = layout


@app.callback(
    Output("board-container", "children"),
    Output("message", "children"),
    Input("reset-button", "n_clicks"),
    Input("pass-button", "n_clicks"),
    [
        Input("button-{}-{}".format(i, j), "n_clicks")
        for i in range(SIZE)
        for j in range(SIZE)
    ],
)
def update_board(n_reset: int, n_pass: int, *args: list[int]) -> tuple[html.Table, str]:
    # コールバックのトリガーを取得
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_board(game.get_board()), dash.no_update
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # 各種ボタンの処理
    if button_id == "reset-button":
        game.reset()
        return create_board(game.get_board()), ""
    elif button_id == "pass-button":
        if len(game.search_place_candidate(user_stone)) != 0:
            return dash.no_update, "You can't pass"
        game.put(Pass(), user_stone)
    else:
        # 押されたボタンの位置を取得
        i, j = map(int, button_id.split("-")[1:])
        place = Place(y=i, x=j)
        # Handle move
        if place not in game.search_place_candidate(user_stone):
            return dash.no_update, "You can't put there"

        game.put(place, user_stone)

    # Agent's move
    agent_move = agent.get_action(
        game.get_board(),
        game.search_place_candidate(ai_stone),
        Turn(ai_stone),
        ai_stone,
    )
    game.put(agent_move, ai_stone)

    # ゲーム終了判定
    if (
        len(game.search_place_candidate(user_stone)) == 0
        and len(game.search_place_candidate(ai_stone)) == 0
    ):
        my_score = game.count_stone(user_stone)
        oppoent_score = game.count_stone(change_stone(user_stone))
        if my_score > oppoent_score:
            message = "You win!"
        elif my_score < oppoent_score:
            message = "You lose..."
        else:
            message = "Draw"
        return create_board(game.get_board()), message

    return create_board(game.get_board()), ""


if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080)
