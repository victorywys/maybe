class BaseRecorder():
    def __init__(self):
        self.record_this_game = True
        
    def init(self, replayer):
        pass
    
    def end_episode(self, main_player_payoff):
        raise NotImplementedError()
    
    def make_selection(self, table, pid, made_action, chi_info=None):
        raise NotImplementedError()
    
    def before_selection(self, table, pid, riichi_step2_tile=None):
        raise NotImplementedError()
    
    def save(self):
        raise NotImplementedError()
    
    
class NullRecorder():
    """
    A recorder that does nothing.
    """
    def __init__(self):
        self.record_this_game = False
        
    def end_episode(self, main_player_payoff):
        pass
    
    def make_selection(self, table, pid, made_action, chi_info=None):
        pass
    
    def before_selection(self, table, pid, riichi_step2_tile=None):
        pass
    
    def save(self):
        pass