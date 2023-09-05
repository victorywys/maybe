PLAYER_OBS_DIM = 93
ORACLE_OBS_DIM = 18
ACTION_DIM = 47
MAHJONG_TILE_TYPES = 34
INIT_POINTS = 25000

# ACTION INDICES
CHILEFT = 34
CHIMIDDLE = 35
CHIRIGHT = 36
PON = 37
ANKAN = 38
MINKAN = 39
KAKAN = 40

RIICHI = 41
RON = 42
TSUMO = 43
PUSH = 44

PASS_RESPONSE = 46
PASS_RIICHI = 45

class MahjongException(Exception):
    pass


class ScoreException(MahjongException):
    def __init__(self, what, paipu, game_order, honba):
        self.info = what
        self.paipu = paipu
        self.ba = game_round(game_order, honba)

    def __str__(self):
        return f"ScoreException: {str(self.info)} {self.paipu} {self.ba}"

    def __repr__(self):
        return self.__str__()


class ActionException(MahjongException):
    def __init__(self, what, paipu, game_order, honba):
        self.info = what
        self.paipu = paipu
        self.ba = game_round(game_order, honba)

    def __str__(self):
        return f"ActionException: {str(self.info)} {self.paipu} {self.ba}"

    def __repr__(self):
        return self.__str__()

# 把副露的5位数解码成具体的东西
def decodem(naru_tiles_int, naru_player_id):
    # 54279 : 4s0s chi 6s
    # 35849 : 6s pon
    # 51275 : chu pon
    # ---------------------------------
    binaries = bin(naru_tiles_int)[2:]

    opened = True

    if len(binaries) < 16:
        binaries = "0" * (16 - len(binaries)) + binaries

    bit2 = int(binaries[-3], 2)
    bit3 = int(binaries[-4], 2)
    bit4 = int(binaries[-5], 2)

    if bit2:
        naru_type = "Chi"

        bit0_1 = int(binaries[-2:], 2)

        if bit0_1 == 3:  # temporally not used
            source = "kamicha"
        elif bit0_1 == 2:
            source = "opposite"
        elif bit0_1 == 1:
            source = "shimocha"
        elif bit0_1 == 0:
            source = "self"

        bit10_15 = int(binaries[:6], 2)
        bit3_4 = int(binaries[-5:-3], 2)
        bit5_6 = int(binaries[-7:-5], 2)
        bit7_8 = int(binaries[-9:-7], 2)

        which_naru = bit10_15 % 3

        source_player_id = (naru_player_id + bit0_1) % 4  # TODO: 包牌

        start_tile_id = int(int(bit10_15 / 3) / 7) * 9 + int(bit10_15 / 3) % 7

        side_tiles_added = [[start_tile_id * 4 + bit3_4, 0], [start_tile_id * 4 + 4 + bit5_6, 0],
                            [start_tile_id * 4 + 8 + bit7_8, 0]]
        # TODO: check aka!
        side_tiles_added[which_naru][1] = 1

        hand_tiles_removed = []
        for kk, ss in enumerate(side_tiles_added):
            if kk != which_naru:
                hand_tiles_removed.append(ss[0])


    else:
        naru_type = "Pon"

        if bit3:

            bit9_15 = int(binaries[:7], 2)

            which_naru = bit9_15 % 3
            pon_tile_id = int(int(bit9_15 / 3))

            side_tiles_added = [[pon_tile_id * 4, 0], [pon_tile_id * 4 + 1, 0], [pon_tile_id * 4 + 2, 0],
                                [pon_tile_id * 4 + 3, 0]]

            bit5_6 = int(binaries[-7:-5], 2)
            which_not_poned = bit5_6

            del side_tiles_added[which_not_poned]

            side_tiles_added[which_naru][1] = 1

            hand_tiles_removed = []
            for kk, ss in enumerate(side_tiles_added):
                if kk != which_naru:
                    hand_tiles_removed.append(ss[0])

        else:  # An-Kan, Min-Kan, Ka-Kan

            bit5_6 = int(binaries[-7:-5], 2)
            which_kan = bit5_6

            if bit4:
                naru_type = "Ka-Kan"
                bit9_15 = int(binaries[:7], 2)

                kan_tile_id = int(bit9_15 / 3)

                side_tiles_added = [[kan_tile_id * 4 + which_kan, 1]]

                hand_tiles_removed = [kan_tile_id * 4 + which_kan]

            else:  # An-Kan or # Min-Kan

                which_naru = naru_tiles_int % 4

                bit8_15 = int(binaries[:8], 2)

                kan_tile = bit8_15
                kan_tile_id = int(kan_tile / 4)
                which_kan = kan_tile % 4

                side_tiles_added = [[kan_tile_id * 4, 0], [kan_tile_id * 4 + 1, 0], [kan_tile_id * 4 + 2, 0],
                                    [kan_tile_id * 4 + 3, 0]]
                if which_naru == 0:
                    naru_type = "An-Kan"
                    hand_tiles_removed = []
                    for kk, ss in enumerate(side_tiles_added):
                        hand_tiles_removed.append(ss[0])
                    opened = False

                else:
                    naru_type = "Min-Kan"
                    side_tiles_added[which_kan][1] = 1

                    hand_tiles_removed = []
                    for kk, ss in enumerate(side_tiles_added):
                        if kk != which_kan:
                            hand_tiles_removed.append(ss[0])

    return side_tiles_added, hand_tiles_removed, naru_type, opened

def game_round(game_order, honba):
    winds = "东南西北"
    chinese_numbers = "一二三四"
    return "此局是{}{}局{}本场".format(winds[game_order // 4],
                                 chinese_numbers[game_order % 4], honba)

def get_tile_from_id(id):
    color = id // 36
    number = (id % 36) // 4 + 1
    color_str = "mpsz"
    return str(number) + color_str[color]


def get_tiles_from_id(tiles):
    ret = ''
    for tile in tiles:
        ret += get_tile_from_id(tile)
    return ret


class logger:
    def __init__(self, fp=None):
        self.fp = fp

    def log(self, *info):
        if self.fp:
            if self.fp == 'stdout':
                print(*info)
            else:
                for s in info:
                    self.fp.write(s)
