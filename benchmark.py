from simulate_game import play_game
import multiprocessing
import pandas as pd
import time
import os
from pathlib import Path


class Benchmark():
    """Benchmark class developped to easily benchmark different model. This means that we want to have an accurate and robust idea of how a model performs compared to other models"""

    def __init__(self, model, count=20, comparing_models = ["greedy_player", "random_player", "team35_with_randomness"], board_sizes=["empty-2x2.txt", "empty-2x3.txt", "empty-3x3.txt", "empty-3x4.txt", "empty-4x4.txt"]):
        self.model = model
        self.count = count  # number of games per board and opponent combination
        self.comparing_models = comparing_models
        self.board_sizes = board_sizes

    def compute_benchmark_results(self):
        """
        Compute a table of results for every model you want to benchmark. Columns are the model to which it has been compared, 
        rows are the board sizes, and the values are the total score of the games played. Compute both as a hierarchical dictionary, 
        pd.Dataframe and save as .csv file.
        """
        all_results = {}
        for opponent in self.comparing_models:
            matchup_dict = {}
            matchup = f"{self.model} - {opponent}"
            for board_name in self.board_sizes:
                board = board_name.split("-")[1].rstrip(".txt")
                print(f"{matchup}; {board}")
                result_of_match = self.play_match(self.model, opponent, f"boards/{board_name}")
                matchup_dict[board] = result_of_match
            all_results[opponent] = matchup_dict
        results_df = pd.DataFrame.from_dict(all_results, orient='columns')

        # Show path and file name were to save the results
        file_path = f"benchmark_results/benchmark_{self.model}_{self.count}count_{len(self.board_sizes)}boards_{len(self.comparing_models)}opps.csv"
        # Make sure we save new results each time, and the file names are updated if already existing
        correct_file_path = self.get_unique_filepath(file_path)
        # Ensure the directory exists
        correct_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Save the DataFrame to the unique file path
        results_df.to_csv(correct_file_path, index=True)
        print(f"Results saved to {correct_file_path}")
        return results_df

    @staticmethod
    def print_score(x: float) -> str:
        """Prints 1 instead of 1.0. Function taken from play_match.py"""
        return '0' if x == 0 else str(x).rstrip('0').rstrip('.')

    def play_match(self, player, opponent, board_sizes, calculation_time=1, verbose=False, warmup=False) -> None:
        """Compute the results of a match between two players, with a specific board size and a number of games. Function largely taken from play_match.py"""
        player_score = 0.0
        opponent_score = 0.0

        for i in range(1, self.count+1):
            # Take turns in starting the game
            print(f'Playing game {i}')
            player_starts = i % 2 == 1
            first = player if player_starts else opponent
            second = opponent if player_starts else player
            result = play_game(board_sizes, first, second, calculation_time, verbose, warmup and i == 1)

            result_line = f'{first} - {second} {self.print_score(result[0])}-{self.print_score(result[1])}\n'
            print(result_line)

            if player_starts:
                player_score += result[0]
                opponent_score += result[1]
            else:
                player_score += result[1]
                opponent_score += result[0]

        result_of_the_match = f'Match result: {player} - {opponent} {self.print_score(player_score)}-{self.print_score(opponent_score)}'
        print(result_of_the_match)

        return f"{player_score} - {opponent_score}"

    @staticmethod
    def get_unique_filepath(filepath):
        """
        Generates a unique file path by appending a suffix if the file already exists.

        Parameters:
        - filepath (str or Path): The desired file path.

        Returns:
        - Path: A unique Path object where the file does not already exist.
        """
        path = Path(filepath)
        if not path.exists():
            return path
        else:
            stem = path.stem  # Filename without suffix (for example .csv or .txt)
            suffix = path.suffix  # File extension, e.g., '.txt'
            parent = path.parent  # Directory path
            i = 1
            while True:
                # Create a new filename with the suffix
                new_name = f"{stem} ({i}){suffix}"
                new_path = parent / new_name
                if not new_path.exists():
                    return new_path
                i += 1

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")  # 'spawn' instead of 'fork' is used in windows

    # To run this benchmark, first create a benchmark object with the desired settings as shown below:
    # benchmark_10_standard = Benchmark('team35_current_best', count=10, comparing_models = ["greedy_player", "team35_with_randomness", "jord_player"], board_sizes=["empty-2x2.txt", "empty-2x3.txt", "empty-3x3.txt", "empty-3x4.txt", "empty-4x4.txt"])
    # benchmark_2_testing = Benchmark('team35_current_best', count=1, comparing_models = ["greedy_player", "team35_with_randomness"], board_sizes=["empty-2x2.txt", "empty-2x3.txt"])
    benchmark_j_v_s = Benchmark('joep_player', count=10, comparing_models = ["stijn_player"], board_sizes=["empty-2x3.txt", "empty-3x3.txt"])


    # All results are then computed and timed
    start = time.perf_counter()
    results = benchmark_j_v_s.compute_benchmark_results()
    end = time.perf_counter()
    print(f"Total elapsed time to compute benchmark: {end - start}")

    # Besides printing the results, the results are also saved in benchmark_results folder
    print(results)