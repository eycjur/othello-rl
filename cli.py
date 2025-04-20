import typer
from tqdm import tqdm

from src.agent import (
    AgentInterface,
    DQNAgent,
    HumanAgent,
    RandomAgent,
    TrainAgentInterface,
)
from src.env import OthelloEnv
from src.utils.enums import Place, Result, Stone
from src.utils.util import Turn

app = typer.Typer()


def learn(
    episode: int,
    size: int,
    my_agent: TrainAgentInterface,
    opponent_agent: AgentInterface,
    render: bool = False,
    batch_size: int = 32,
    is_test: bool = False,
    model_sync_frequency: int = 25,
) -> None:
    env = OthelloEnv(size)

    list_result = []
    for i in tqdm(range(episode), desc="episode", dynamic_ncols=True):
        mystone = [Stone.BLACK, Stone.WHITE][i % 2]
        observation = env.reset(mystone=mystone, opponent_agent=opponent_agent)
        done = False

        while not done:
            action_candidate = env.search_action_candidate(env.turn)
            action = my_agent.get_action(
                observation, action_candidate, env.turn, mystone
            )
            observation_next, reward, done, _ = env.step(action)

            if not is_test and isinstance(action, Place):
                my_agent.remember(
                    state=observation,
                    action=action,
                    reward=reward,
                    next_state=observation_next,
                    next_action_candidates=action_candidate,
                    turn=Turn(mystone),
                    done=done,
                )

                if batch_size < len(my_agent.memory):
                    my_agent.replay(batch_size)

            if render:
                env.render()
            action_candidate = env.search_action_candidate(env.turn)

            observation = observation_next
        list_result.append(env.calc_result())

        if not is_test and (i + 1) % model_sync_frequency == 0:
            print(
                f"{i}回目: "
                f"{list_result[- model_sync_frequency:].count(Result.WIN)}勝, "
                f"{list_result[- model_sync_frequency:].count(Result.LOSE)}敗, "
                f"{list_result[- model_sync_frequency:].count(Result.DRAW)}引き分け"
            )
            my_agent.plot_history()
            my_agent.sync_model(opponent_agent)
            my_agent.save("output/agent.pkl")

    if not is_test:
        my_agent.plot_history()
        my_agent.save("output/agent.pkl")

    print(f"{list_result.count(Result.WIN)}勝")
    print(f"{list_result.count(Result.LOSE)}敗")
    print(f"{list_result.count(Result.DRAW)}引き分け")


@app.command()
def train(
    episode: int = typer.Option(300, "-e", "--episode", help="学習回数"),
    eposode_test: int = typer.Option(300, "-t", "--episode-test", help="評価回数"),
) -> None:
    size = 6
    my_agent = DQNAgent(shape=(size, size))
    opponent_agent_for_train = DQNAgent(shape=(size, size))

    learn(episode, size, my_agent, opponent_agent_for_train, False, is_test=False)
    my_agent.test()
    opponent_agent_for_valid = RandomAgent()
    learn(eposode_test, size, my_agent, opponent_agent_for_valid, False, is_test=True)


@app.command()
def test() -> None:
    size = 6
    my_agent = TrainAgentInterface.load("output/agent.pkl")
    my_agent.test()
    opponent_agent = HumanAgent()

    learn(30, size, my_agent, opponent_agent, False, is_test=True)


if __name__ == "__main__":
    app()
