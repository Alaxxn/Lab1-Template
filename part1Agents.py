from model import (
    Location,
    Portal,
    EmptyEntity,
    Wizard,
    Goblin,
    Crystal,
    WizardMoves,
    GoblinMoves,
    GameAction,
    GameState,
)
from agents import WizardSearchAgent
import heapq
from dataclasses import dataclass

class WizardDFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)

    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    # (Node, path_to_node)
    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc
        
    def next_search_expansion(self) -> GameState | None:
        current_search_state = self.search_stack.pop()
        return self.search_to_game(current_search_state) # Expand current state

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        """
        source -> Game state passed by next_search_expansion
        action -> Direction of wizard move 
        target -> Resulting game state after action
        """

        source_search = self.game_to_search(source)
        target_search = self.game_to_search(target)
        path_to_target = self.paths[source_search] + [action]

        if (target_search in self.paths): return # skip states already visited
        
        self.search_stack.append(target_search) # Update Frontier
        self.paths[target_search] = path_to_target # Update Visited 

        if (self.is_goal(target_search)): # Goal Found
            moves = path_to_target.copy()
            moves.reverse() # for game engine, uses pop() for next move
            self.plan = moves # Ends Search

class WizardBFS(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, list[WizardMoves]] = {}
    search_stack: list[SearchState] = []
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = []
        self.search_stack = [initial_search_state]

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def next_search_expansion(self) -> GameState | None:
        current_search_state = self.search_stack.pop(0)
        return self.search_to_game(current_search_state)

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        source_search = self.game_to_search(source)
        target_search = self.game_to_search(target)
        path_to_target = self.paths[source_search] + [action]

        if (target_search in self.paths): return # skip states already visited
        
        self.search_stack.append(target_search) # Update Frontier
        self.paths[target_search] = path_to_target # Update Visited 

        if (self.is_goal(target_search)): # Goal Found
            moves = path_to_target.copy()
            moves.reverse() # for game engine, uses pop() for next move
            self.plan = moves # Ends Search

class WizardAstar(WizardSearchAgent):
    @dataclass(eq=True, frozen=True, order=True)
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    paths: dict[SearchState, tuple[float, list[WizardMoves]]] = {}
    search_pq: list[tuple[float, SearchState]] = [] # Frontier
    initial_game_state: GameState

    def search_to_game(self, search_state: SearchState) -> GameState:
        initial_wizard_loc = self.initial_game_state.active_entity_location
        initial_wizard = self.initial_game_state.get_active_entity()

        new_game_state = (
            self.initial_game_state.replace_entity(
                initial_wizard_loc.row, initial_wizard_loc.col, EmptyEntity()
            )
            .replace_entity(
                search_state.wizard_loc.row, search_state.wizard_loc.col, initial_wizard
            )
            .replace_active_entity_location(search_state.wizard_loc)
        )

        return new_game_state

    def game_to_search(self, game_state: GameState) -> SearchState:
        wizard_loc = game_state.active_entity_location
        portal_loc = game_state.get_all_tile_locations(Portal)[0]
        return self.SearchState(wizard_loc, portal_loc)

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def start_search(self, game_state: GameState):
        self.initial_game_state = game_state

        initial_search_state = self.game_to_search(game_state)
        self.paths = {}
        self.paths[initial_search_state] = 0, []
        self.search_pq = [(0, initial_search_state)]
        heapq.heapify(self.search_pq)

    def is_goal(self, state: SearchState) -> bool:
        return state.wizard_loc == state.portal_loc

    def cost(self, source: GameState, target: GameState, action: WizardMoves) -> float:
        return 1

    def heuristic(self, n: GameState) -> float:
        """ Manhattan Distance to the portal """
        curr_search_state = self.game_to_search(n)

        pos = curr_search_state.wizard_loc
        goal = curr_search_state.portal_loc

        x_diff = abs(pos.row - goal.row)
        y_diff = abs(pos.col - goal.col)

        return x_diff + y_diff
    
    def next_search_expansion(self) -> GameState | None:

        curr_state = heapq.heappop(self.search_pq)[1]

        if (self.is_goal(curr_state)): # Goal Found
            path_to_curr = self.paths[curr_state][1]
            moves = path_to_curr.copy()
            moves.reverse()
            self.plan = moves
            return

        return self.search_to_game(curr_state)

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        cost_index = 0
        path_to_state_index = 1
        
        source_search = self.game_to_search(source)
        target_search = self.game_to_search(target)
        source_cost, source_path = self.paths[source_search]
        target_path = source_path + [action]

        if (target_search in self.paths): return
        
        # g(n)
        target_true_cost = source_cost + self.cost(source, target, action)
        # f(n) = g(n) + h(n)
        step_combined_cost = target_true_cost + self.heuristic(target)

        # Update Frontier
        heapq.heappush(self.search_pq, (step_combined_cost, target_search))
        
        # Update Visited
        cost_to_source, path_to_source = self.paths[source_search]
        self.paths[target_search] = (target_true_cost, path_to_source + [action])

class CrystalSearchWizard(WizardSearchAgent):
    # TODO: YOUR CODE HERE

    def __init__(self, initial_state: GameState):
        self.start_search(initial_state)

    def next_search_expansion(self) -> GameState | None:
        # TODO YOUR CODE HEREs
        raise NotImplementedError

    def process_search_expansion(
        self, source: GameState, target: GameState, action: WizardMoves
    ) -> None:
        # TODO YOUR CODE HERE
        raise NotImplementedError


class SuboptimalCrystalSearchWizard(CrystalSearchWizard):
    class SearchState:
        wizard_loc: Location
        portal_loc: Location

    def heuristic(self, target: SearchState) -> float:
        # TODO YOUR CODE HERE
        raise NotImplementedError
