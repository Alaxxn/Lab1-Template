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

    def manhattan_distance(self, pos : Location, target: Location) -> int:
        """ Manhattan Distance to the target """
        x_diff = abs(pos.row - target.row)
        y_diff = abs(pos.col - target.col)
        return x_diff + y_diff
    
    def evaluation(self, state: GameState) -> float:

        ""
        # Want to maximize this eval at each step
        portal_loc = state.get_all_tile_locations(Portal)[0]
        goblin_loc = state.get_all_entity_locations(Goblin)[0]
        wiz_loc = state.get_all_entity_locations(Wizard)[0]
        
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

    def is_terminal(self, state: GameState) -> bool:

        wizard_locs = state.get_all_entity_locations(Wizard)
        if len(wizard_locs) == 0: # Wizard Lost
            return True

        wizard_loc = wizard_locs[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]
        if wizard_loc == portal_loc: # Wizard Won
            return True

        return False

    def react(self, state: GameState) -> WizardMoves:
        best_move, gain = self.minimax(state, self.max_depth)
        return  best_move if best_move is not None else WizardMoves.STAY


    def minimax(self, state: GameState, depth: int):

        if (self.is_terminal(state)): # Game Ended
            wiz_locs = state.get_all_entity_locations(Wizard)
            if (len(wiz_locs) == 0): # Wizard never reached the goal
                return (None, float('-inf'))
            else: # Wizard WON
                return (None, float('inf'))

        if (depth == 0): # Evaluate Leaf
            return (None, self.evaluation(state))        
        
        move = None
        active = state.get_active_entity()

        if isinstance(active, Wizard): #Maximizer
            best_eval = float ('-inf')
            for action, result in self.get_successors(state):
                val = self.minimax(result, depth - 1)[1]
                if (val > best_eval):
                    move = action
                    best_eval = val              
            return (move, best_eval)
        
        else : # Minimizer
            best_eval = float('inf')
            for action, result in self.get_successors(state):
                val = self.minimax(result, depth)[1] # Does not affect depth
                if (val < best_eval):
                    move = action
                    best_eval = val
            return (move, best_eval)


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
