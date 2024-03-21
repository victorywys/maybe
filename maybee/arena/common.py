import numpy as np
import pymahjong as pm
from pymahjong import Yaku

UNICODE_TILES = """
    🀇 🀈 🀉 🀊 🀋 🀌 🀍 🀎 🀏
    🀙 🀚 🀛 🀜 🀝 🀞 🀟 🀠 🀡
    🀐 🀑 🀒 🀓 🀔 🀕 🀖 🀗 🀘
    🀀 🀁 🀂 🀃
    🀆 🀅 🀄
    🀋 🀝 🀔
""".split()
ACTIONS = ["摸牌", "摸杠牌", "手切", "摸切", "吃L", "吃M", "吃R", "碰", "明杠", "暗杠", "加杠", "手切立直", "摸切立直", "立直通过", "自家", "下家", "对家", "上家"]

tile_to_human = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "dong", "nan", "xi", "bei", "bai", "fa", "zhong",
    "0m", "0p", "0s",
]
ju_to_human = ["东一局", "东二局", "东三局", "东四局", "南一局", "南二局", "南三局", "南四局", "西一局", "西二局", "西三局", "西四局", "北一局", "北二局", "北三局", "北四局"]
human_to_tile = {
    tile: i for i, tile in enumerate(tile_to_human)
}

action_v2_to_human = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "East (1z)", "South (2z)", "West (3z)", "North (4z)", "Haku (5z)", "Hatsu (6z)", "Chu (7z)", "0m (red 5m)", "0p (red 5p)",
    "0s (red 5s)", "CHILEFT", "CHIMIDDLE", "CHIRIGHT", "CHILEFT_RED", "CHIMIDDLE_RED", "CHIRIGHT_RED", "PON", "PON_RED",
    "ANKAN", "MINKAN", "KAKAN", "RIICHI", "RON", "TSUMO", "PUSH", "PASS_RIICHI", "PASS_RESPONSE"]

action_v2_to_human_chinese = [
    "一万", "二万", "三万", "四万", "五万", "六万", "七万", "八万", "九万",
    "一筒", "二筒", "三筒", "四筒", "五筒", "六筒", "七筒", "八筒", "九筒",
    "一索", "二索", "三索", "四索", "五索", "六索", "七索", "八索", "九索",
    "东风", "南风", "西风", "北风", "白板", "发财", "中", "赤五万", "赤五筒",
    "赤五索", "吃左", "吃中", "吃右", "红吃左", "红吃中", "红吃右", "碰", "红碰",
    "暗杠", "明杠", "加杠", "立直", "荣和", "自摸", "推九九", "不立直", "过"]

# human_to_tile["0m"] = 4
# human_to_tile["0p"] = 13
# human_to_tile["0s"] = 22

feng_to_human = ["东", "南", "西", "北"]
human_to_action = {
    "cl": 34,
    "cm": 35,
    "cr": 36,
    "pon": 37,
    "ag": 38,
    "mg": 39,
    "jg": 40,
    "ron": 42,
    "zm": 43,
    "99": 44,
    "p": 46,
}
action_to_human = {
    i: a for a, i in human_to_action.items()
}
    
tile_to_tenhou = list(range(11, 20)) + list(range(21, 30)) + list(range(31, 40)) + list(range(41, 48)) + [51, 52, 53]
tile_name_to_tenhou = {
    "1m": 11,
    "2m": 12,
    "3m": 13,
    "4m": 14,
    "5m": 15,
    "6m": 16,
    "7m": 17,
    "8m": 18,
    "9m": 19,
    "1p": 21,
    "2p": 22,
    "3p": 23, 
    "4p": 24,
    "5p": 25,
    "6p": 26,
    "7p": 27,
    "8p": 28,
    "9p": 29,
    "1s": 31,
    "2s": 32,
    "3s": 33,
    "4s": 34,
    "5s": 35,
    "6s": 36,
    "7s": 37,
    "8s": 38,
    "9s": 39,
    "1z": 41,
    "2z": 42,
    "3z": 43,
    "4z": 44,
    "5z": 45,
    "6z": 46,
    "7z": 47,
    "0m": 51,
    "0p": 52,
    "0s": 53,
}

yaku_to_tenhou = {		
    Yaku.Riichi: "立直(1飜)",
	Yaku.Tanyao: "断幺九(1飜)",
    Yaku.Pinfu: "平和(1飜)",
    Yaku.Yiipeikou: "一盃口(1飜)",
    Yaku.Menzentsumo: "門前清自摸和(1飜)",
    Yaku.SelfWind_East: "自風 東(1飜)",
    Yaku.SelfWind_South: "自風 南(1飜)",
    Yaku.SelfWind_West: "自風 西(1飜)",
    Yaku.SelfWind_North: "自風 北(1飜)",
    Yaku.GameWind_East: "場風 東(1飜)",
    Yaku.GameWind_South: "場風 南(1飜)",
    Yaku.GameWind_West: "場風 西(1飜)",
    Yaku.GameWind_North: "場風 北(1飜)",
    Yaku.Yakuhai_haku: "役牌 白(1飜)",
    Yaku.Yakuhai_hatsu: "役牌 發(1飜)",
    Yaku.Yakuhai_chu: "役牌 中(1飜)",
    Yaku.Chankan: "槍槓(1飜)",
    Yaku.Rinshankaihou: "嶺上開花(1飜)",
    Yaku.Haitiraoyue: "海底摸月(1飜)",
    Yaku.Houtiraoyui: "河底撈魚(1飜)",
    Yaku.Ippatsu: "一発(1飜)",
    Yaku.Chantai_: "混全帯幺九(1飜)",
    Yaku.Ikkitsuukan_: "一気通貫(1飜)",
    Yaku.Sanshokudoujun_: "三色同順(1飜)",
    Yaku.DoubleRiichi: "両立直(2飜)",
    Yaku.Sanshokudoukou: "三色同刻(2飜)",
    Yaku.Sankantsu: "三槓子(2飜)",
    Yaku.Toitoi: "対々和(2飜)",
    Yaku.Sanankou: "三暗刻(2飜)",
    Yaku.Shosangen: "小三元(2飜)",
    Yaku.Honrotou: "混老頭(2飜)",
    Yaku.Chitoitsu: "七対子(2飜)",
    Yaku.Chantai: "混全帯幺九(2飜)",
    Yaku.Ikkitsuukan: "一気通貫(2飜)",
    Yaku.Sanshokudoujun: "三色同順(2飜)",
    Yaku.Junchan_: "純全帯幺九(2飜)",
    Yaku.Honitsu_: "混一色(2飜)",
    Yaku.Ryanpeikou: "二盃口(3飜)",
    Yaku.Junchan: "純全帯幺九(3飜)",
    Yaku.Honitsu: "混一色(3飜)",
    Yaku.Chinitsu_: "清一色(5飜)",
    Yaku.Chinitsu: "清一色(6飜)",
    Yaku.Tenho: "天和(役満)",
    Yaku.Chiiho: "地和(役満)",
    Yaku.Daisangen: "大三元(役満)",
    Yaku.Suuanko: "四暗刻(役満)",
    Yaku.Tsuuiisou: "字一色(役満)",
    Yaku.Ryuiisou: "緑一色(役満)",
    Yaku.Chinroutou: "清老頭(役満)",
    Yaku.Koukushimusou: "国士無双(役満)",
    Yaku.Shosushi: "小四喜(役満)",
    Yaku.Suukantsu: "四槓子(役満)",
    Yaku.Churenpoutou: "九蓮宝燈(役満)",
    Yaku.SuuankoTanki: "四暗刻単騎(役満)",
    Yaku.Koukushimusou_13: "国士無双１３面(役満)",
    Yaku.Pure_Churenpoutou: "純正九蓮宝燈(役満)",
    Yaku.Daisushi: "大四喜(役満)",
    Yaku.Dora: "ドラ",
    Yaku.UraDora: "裏ドラ",
    Yaku.AkaDora: "赤ドラ",
}

def get_base_tile(tile):
    if tile == 34:
        return 4
    elif tile == 35:
        return 13
    elif tile == 36:
        return 22
    else:
        return tile

def render_global_info(global_info):
    explains = ["局数", "最终局", "本场数", "场供数", "自风牌", "场风牌", "自家点数", "下家点数", "对面点数", "上家点数", "自家一发", "下家一发", "对面一发", "上家一发", "剩余牌数"]
    for i, e in enumerate(explains):
        print(e + ":" + str(global_info[i]))

def render_encoding_record(record):
    a = np.argwhere(np.array(record)).reshape([-1])
    action_strs = []
    for i in reversed(a):
        if i < 37:
            action_strs.append(tile_to_human[i])
        else:
            action_strs.append(ACTIONS[i - 37])
    print(" ".join(action_strs))

def render_encoding_self_info(self_info):
    # 0-3

    hands = list(np.argwhere(self_info[:, 0]).reshape([-1])) + list(np.argwhere(self_info[:, 1]).reshape([-1])) + list(np.argwhere(self_info[:, 2]).reshape([-1])) + list(np.argwhere(self_info[:, 3]).reshape([-1]))
    hands.sort()
    print("-------- Hand -------------")
    print("".join([UNICODE_TILES[i] for i in hands]))

    hand_akas = list(np.argwhere(self_info[:, 6]).reshape([-1]))
    print("-------- Aka -------------")
    print("".join([UNICODE_TILES[i] for i in hand_akas]))


    doras = list(np.argwhere(self_info[:, 4]).reshape([-1]))
    print("-------- Dora -------------")
    print("".join([UNICODE_TILES[i] for i in doras]))

    dora_indicators = list(np.argwhere(self_info[:, 5]).reshape([-1]))
    print("-------- Dora Indicator -------------")
    print("".join([UNICODE_TILES[i] for i in dora_indicators]))

    changfengs = list(np.argwhere(self_info[:, 7]).reshape([-1]))
    print("-------- Game Wind -------------")
    print("".join([UNICODE_TILES[i] for i in changfengs]))

    zifengs = list(np.argwhere(self_info[:, 8]).reshape([-1]))
    print("-------- Self Wind -------------")
    print("".join([UNICODE_TILES[i] for i in zifengs]))


    zimopai = list(np.argwhere(self_info[:, 9]).reshape([-1]))
    print("-------- Tsumo Tiles -------------")
    print("".join([UNICODE_TILES[i] for i in zimopai]))

    tmp = list(np.argwhere(self_info[:, 10]).reshape([-1]))
    print("-------- Self Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))

    tmp = list(np.argwhere(self_info[:, 11]).reshape([-1]))
    print("-------- Next Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))

    tmp = list(np.argwhere(self_info[:, 12]).reshape([-1]))
    print("-------- Opposite Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))

    tmp = list(np.argwhere(self_info[:, 13]).reshape([-1]))
    print("-------- Previous Discarded -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))


    tmp = list(np.argwhere(self_info[:, 14]).reshape([-1])) + list(np.argwhere(self_info[:, 15]).reshape([-1])) + list(np.argwhere(self_info[:, 16]).reshape([-1])) + list(np.argwhere(self_info[:, 17]).reshape([-1]))
    tmp.sort()
    print("-------- Disclosed Tiles -------------")
    print("".join([UNICODE_TILES[i] for i in tmp]))