from .Base import PLAYER, BasePlayer
import numpy as np

from arena.common import tile_to_human, human_to_tile, action_to_human, human_to_action,\
    ju_to_human, feng_to_human, \
    render_encoding_record



@PLAYER.register_module(name="human")
class HumanPlayer(BasePlayer):
    def __init__(
        self, 
        name: str
    ):
        super.__init__(name)
        self.deside_riichi = False
        self.printed_line = 0

    def simplify_hand(
        self,
        hand,
    ):
        m = []
        p = []
        s = []
        z = []
        for tile in hand:
            if len(tile) == 2 and tile[1] == "m":
                m.append(tile[0])
            elif len(tile) == 2 and tile[1] == "p":
                p.append(tile[0])
            elif len(tile) == 2 and tile[1] == "s":
                s.append(tile[0])
            else:
                z.append(tile)
        m.sort()
        p.sort()
        s.sort()
        if '0' in m:
            m_repr = "".join(list(filter(lambda x: '1' <= x <= '5', m)) + ['0'] + list(filter(lambda x: '6' <= x <= '9', m)))
        else:
            m_repr = "".join(m)
        if '0' in p:
            p_repr = "".join(list(filter(lambda x: '1' <= x <= '5', p)) + ['0'] + list(filter(lambda x: '6' <= x <= '9', p)))
        else:
            p_repr = "".join(p)
        if '0' in s:
            s_repr = "".join(list(filter(lambda x: '1' <= x <= '5', s)) + ['0'] + list(filter(lambda x: '6' <= x <= '9', s)))
        else:
            s_repr = "".join(s)
        z_repr = " ".join(z)
        repr = ""
        if len(m_repr) > 0:
            repr += m_repr + "m"
        if len(p_repr) > 0:
            repr += p_repr + "p"
        if len(s_repr) > 0:
            repr += s_repr + "s"
        if len(z_repr) > 0:
            if len(repr) > 0:
                repr += " "
            repr += z_repr
        return repr

    def print_valid_actions(self, valid_actions):
        for a in valid_actions:
            if a < 37:
                print(tile_to_human[a])
            else:
                print(action_to_human[a])

    def play(
        self, 
        self_info: np.ndarray,
        record_info: np.ndarray,
        global_info: np.ndarray,
        valid_actions_mask: np.ndarray,
    ):
        print("=====================================")
        ju = global_info[0]
        benchang = global_info[2]
        changgong = global_info[3]
        zifeng = global_info[4]
        print(f"{ju_to_human[ju]} {benchang}本场 {changgong*1000}供托 自风{feng_to_human[zifeng]}")
        dora_indicators = list(np.argwhere(self_info[:, 5]).reshape([-1]))
        print("宝牌指示牌: ", " ".join([tile_to_human[tile] for tile in dora_indicators]))
        print("点数：")
        print(f"     {global_info[8]}     ")
        print(f"{global_info[9]}        {global_info[7]}")
        print(f"     {global_info[6]}     ")

        if len(record_info) < self.printed_line:
            self.printed_line = 0
        for i in range(self.printed_line, len(record_info)):    
            a = np.argwhere(np.array(record_info[i])).reshape([-1])
            if 37 not in a and 38 not in a:
                render_encoding_record(record_info[i])

        self.printed_line = len(record_info)

        hands = list(np.argwhere(self_info[:, 0]).reshape([-1])) + list(np.argwhere(self_info[:, 1]).reshape([-1])) + list(np.argwhere(self_info[:, 2]).reshape([-1])) + list(np.argwhere(self_info[:, 3]).reshape([-1]))
        hands.sort()

        zimopai = list(np.argwhere(self_info[:, 9]).reshape([-1]))
        if len(zimopai) > 0:
            hands.remove(zimopai[0])
            zimopai = zimopai[0]

        hand_akas = list(np.argwhere(self_info[:, 6]).reshape([-1]))
        if 4 in hand_akas:
            if 4 in hands:
                hands.remove(4)
                hands.append(34)
            else:
                zimopai = 34
        if 13 in hand_akas:
            if 13 in hands:
                hands.remove(13)
                hands.append(35)
            else:
                zimopai = 35
        if 22 in hand_akas:
            if 22 in hands:
                hands.remove(22)
                hands.append(36)
            else:
                zimopai = 36

        hands = [tile_to_human[tile] for tile in hands]
        if isinstance(zimopai, np.int64):
            print("自家手牌: ", self.simplify_hand(hands + [tile_to_human[zimopai]]))
        else:
            print("自家手牌: ", self.simplify_hand(hands))

        if record_info.shape[0] > 0:
            last_record = record_info[-1]
            last_action = np.argwhere(np.array(last_record)).reshape([-1])
        else:
            last_action = [zimopai, 37, 51]
        
        valid_actions = np.argwhere(valid_actions_mask).reshape([-1]).tolist()
        if 41 in valid_actions and 45 in valid_actions: # 立直
            if self.deside_riichi:
                return 41
            else:
                return 45
        while True:
            if 37 in last_action or 38 in last_action: # 摸牌， 摸杠牌
                print("自家自摸牌: ", tile_to_human[zimopai])
            a_raw = input("Please input your action: ")
            if a_raw == "chi":
                if 34 in valid_actions and 35 not in valid_actions and 36 not in valid_actions:
                    a = 34
                elif 34 not in valid_actions and 35 in valid_actions and 36 not in valid_actions:
                    a = 35
                elif 34 not in valid_actions and 35 not in valid_actions and 36 in valid_actions:
                    a = 36
                elif 34 not in valid_actions and 35 not in valid_actions and 36 not in valid_actions:
                    print("chi is not a valid action")
                    self.print_valid_actions(valid_actions)
                    continue
                else:
                    print("Multiple possibility for chi, please specify.")
                    self.print_valid_actions(valid_actions)
                    continue
            
            elif a_raw in human_to_action:
                a = human_to_action[a_raw]
            elif a_raw[0] == "r" and a_raw[1:] in human_to_tile:
                self.deside_riichi = True
                a = human_to_tile[a_raw[1:]]
            elif a_raw in human_to_tile:
                self.deside_riichi = False
                a = human_to_tile[a_raw]
            else:
                print("Invalid action")
                self.print_valid_actions(valid_actions)
                continue
            
            if a not in valid_actions:
                print("Invalid action")
                self.print_valid_actions(valid_actions)
                continue
                
            return a