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

        # Evaluatet terminal states
        wizard_locs = state.get_all_entity_locations(Wizard)
        if len(wizard_locs) == 0: # Wizard Lost
            return float('-inf')

        wizard_loc = wizard_locs[0]
        portal_loc = state.get_all_tile_locations(Portal)[0]
        if wizard_loc == portal_loc: # Wizard Won
            return float('inf')
        
        # Non-terminal Evals
        portal_loc = state.get_all_tile_locations(Portal)[0]
        goblin_locs = state.get_all_entity_locations(Goblin)
        wiz_loc = state.get_all_entity_locations(Wizard)[0]
        
        if len(goblin_locs) > 0: # Only worry about nearest goblin
            nearest_goblin = float('inf')
            for loc in goblin_locs:
                dist = self.manhattan_distance(loc, wiz_loc)
                if (dist < nearest_goblin):
                    nearest_goblin = dist
        else:
            nearest_goblin = 999 #
        
        portal_dist = self.manhattan_distance(wiz_loc, portal_loc)

        score = 0
        score -= 10 * portal_dist # Want to reward movement toward goal
        score += 2 * nearest_goblin # Want to penalize getting near goblin

        if nearest_goblin <= 1: # RUN AWAY
            score -= 100
        elif nearest_goblin == 2: # Move Back
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

        best_eval = float('-inf')
        best_move = WizardMoves.STAY 

        for action, next_state in self.get_successors(state):
  
            if (action == WizardMoves.STAY):
                continue

            action_eval = self.minimax(next_state, self.max_depth - 1)
            if action_eval > best_eval:
                best_eval = action_eval
                best_move = action

        print(best_move, best_eval )
        return  best_move


    def minimax(self, state: GameState, depth: int):
        """ returns the best possible evaluation for the active enity"""

        if (self.is_terminal(state) or (depth == 0)):
            return self.evaluation(state)

        active = state.get_active_entity()    

        if isinstance(active, Wizard):
            return self.wizard_maximizer(state, depth)
        else :
            return self.goblin_minimizer(state, depth)


    def goblin_minimizer(self, state, depth) -> float: 

        if (self.is_terminal(state)) or (depth == 0):
            return self.evaluation(state)
        
        v = float('inf')
        for _, next_state in self.get_successors(state):
            next_entity = next_state.get_active_entity()
            if isinstance(next_entity, Wizard):
                v = min(v, self.wizard_maximizer(next_state, depth - 1))
            else:
                v = min(v, self.goblin_minimizer(next_state, depth))

        return v 

    def wizard_maximizer(self, state: GameState, depth: int) -> float:

        if (self.is_terminal(state)) or (depth == 0):
            return self.evaluation(state)
        
        v = float('-inf')
        for _, next_state in self.get_successors(state):
            v = max(v, self.goblin_minimizer(next_state, depth))
        
        return v


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
