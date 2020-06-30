import numpy as np
from hangmanPlayer import HangmanPlayer
from hangmanPlayerV2 import HangmanPlayerV2
from hangmanPlayerV3 import HangmanPlayerV3


class HangmanServer:
    def __init__(self, player):
        self.player = player
        self.test_words = []

    @staticmethod
    def read_test_words():
        with open("words_alpha_test_unique.txt") as f:
            words = f.read().split('\n')

        return words[:-1]

    @staticmethod
    def data_iter(words):
        for word in words:
            _, answer = word.split(',')
            question = '#' * len(answer)
            yield question, answer

    def run(self):
        test_words = self.read_test_words()
        np.random.shuffle(test_words)
        test_words = test_words[:1000]
        qa_pair = self.data_iter(test_words)
        success = total = 0
        success_rate = 0
        print(f"Total Game Number: {len(test_words)}")
        for question, answer in qa_pair:
            self.player.new_game()
            tries = 6
            success_rate = 0 if total == 0 else success / total
            print("=" * 20, "Game %d" % (total + 1), '=' * 20, "Success Rate: %.2f" % success_rate)
            # if (total + 1) % 100 == 0:
            #     print(total + 1)
            print('provided question: ', " ".join(question))
            while '#' in question and tries > 0:
                guess = self.player.guess(question)
                question_lst = []
                for q_l, a_l in zip(question, answer):
                    if q_l == '#':
                        if a_l == guess:
                            question_lst.append(a_l)
                        else:
                            question_lst.append(q_l)
                    else:
                        question_lst.append(q_l)
                question = "".join(question_lst)
                if guess not in answer:
                    tries -= 1
                print("provided question: ", " ".join(question), "your guess: %s" % guess, "left tries: %d" % tries, 'answer: %s' % answer)

            if '#' not in question:
                success += 1
            total += 1

        print(f"{success} success out of {total} tries, rate: {success / total:.4f}")


if __name__ == "__main__":
    player = HangmanPlayerV3()
    server = HangmanServer(player)

    server.run()
