from model import (
    Location,
    Portal,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import ReasoningWizard
from dataclasses import dataclass
    

class WizardGreedy(ReasoningWizard):

    def manhattan_distance(self, pos : Location, target: Location) -> int:
        """ Manhattan Distance to the target """
        x_diff = abs(pos.row - target.row)
        y_diff = abs(pos.col - target.col)
        return x_diff + y_diff
    

    def evaluation(self, state: GameState) -> float:

        # Want to maximize this eval at each step

        portal_loc = state.get_all_tile_locations(Portal)[0]
        goblin_loc = state.get_all_entity_locations(Goblin)[0]
        wiz_loc = state.get_all_entity_locations(Wizard)[0]

        if (wiz_loc == portal_loc): # This is our goal
            return float('inf')
        
        portal_dist = self.manhattan_distance(wiz_loc, portal_loc)
        goblin_dist = self.manhattan_distance(wiz_loc, goblin_loc)
        
        score = 0
        score -= 10 * portal_dist # Want to reward movement toward goal
        score += 2 * goblin_dist # Want to penalize getting near goblin

        if goblin_dist <= 1: # RUN AWAY
            score -= 100
        elif goblin_dist == 2: # Move Back
            score -= 20
        
        return score



class WizardMiniMax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        raise NotImplementedError


    def minimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        raise NotImplementedError


class WizardAlphaBeta(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        raise NotImplementedError


    def alpha_beta_minimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        raise NotImplementedError




class WizardExpectimax(ReasoningWizard):
    max_depth: int = 2

    def evaluation(self, state: GameState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def is_terminal(self, state: GameState) -> bool:
        # TODO YOUR CODE HERE
        raise NotImplementedError

    def react(self, state: GameState) -> WizardMoves:
        # TODO YOUR CODE HERE
        raise NotImplementedError


    def expectimax(self, state: GameState, depth: int):
        # TODO YOUR CODE HERE
        raise NotImplementedError
