from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0}


def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda : 0)
    
    STARFRUIT_cache = []
    STARFRUIT_dim = 3
    
    buy_orchids = False
    sell_orchids = False
    close_orchids = False
    # last_dg_price = 0
    start_orchids = 0
    first_orchids = 0

    #check use
    orchids_cache = []
    orchids_dim = 3

    steps = 0
    
    halflife_diff = 5
    alpha_diff = 1 - np.exp(-np.log(2)/halflife_diff)

    halflife_price = 5
    alpha_price = 1 - np.exp(-np.log(2)/halflife_price)

    halflife_price_dip = 20
    alpha_price_dip = 1 - np.exp(-np.log(2)/halflife_price_dip)
    
    begin_diff_dip = -INF
    begin_diff_bag = -INF
    begin_bag_price = -INF
    begin_dip_price = -INF

    def compute_orchids(self, observations, order_depth, timestamp):
        orders = {'ORCHIDS' : []}
        prods = ['ORCHIDS']

        # check variables
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}
       
        for p in prods:
            osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
            obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))

            best_sell[p] = next(iter(osell[p]))
            best_buy[p] = next(iter(obuy[p]))

            worst_sell[p] = next(reversed(osell[p]))
            worst_buy[p] = next(reversed(obuy[p]))

            mid_price[p] = (best_sell[p] + best_buy[p])/2
            vol_buy[p], vol_sell[p] = 0, 0
            for price, vol in obuy[p].items():
                vol_buy[p] += vol 
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 

        if timestamp == 0:
            self.start_orchids = mid_price['ORCHIDS']
        if timestamp == 350*1000:
            self.first_orchids = mid_price['ORCHIDS']
            self.buy_orchids = True
        if timestamp == 500*1000:
            self.sell_orchids = True
        if timestamp == 750*1000: 
            if self.first_orchids != 0 and self.start_orchids != 0 and self.first_orchids > self.start_orchids:
                self.buy_orchids = True
            elif self.first_orchids == 0 or self.start_orchids == 0:
                self.close_orchids = True

        if int(round(self.person_position['Olivia']['ORCHIDS'])) > 0:
            self.buy_orchids = True
            self.sell_orchids = False
        if int(round(self.person_position['Olivia']['ORCHIDS'])) < 0:
            self.sell_orchids = True
            self.buy_orchids = False

        if self.buy_orchids and self.position['ORCHIDS'] == self.POSITION_LIMIT['ORCHIDS']:
            self.buy_orchids = False
        if self.sell_orchids and self.position['ORCHIDS'] == -self.POSITION_LIMIT['ORCHIDS']:
            self.sell_orchids = False
        if self.close_orchids and self.position['ORCHIDS'] == 0:
            self.close_orchids = False

        if self.buy_orchids:
            vol = self.POSITION_LIMIT['ORCHIDS'] - self.position['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', best_sell['ORCHIDS'], vol))
        if self.sell_orchids:
            vol = self.position['ORCHIDS'] + self.POSITION_LIMIT['ORCHIDS']
            orders['ORCHIDS'].append(Order('ORCHIDS', best_buy['ORCHIDS'], -vol))
        if self.close_orchids:
            vol = -self.position['ORCHIDS']
            if vol < 0:
                orders['ORCHIDS'].append(Order('ORCHIDS', best_buy['ORCHIDS'], vol)) 
            else:
                orders['ORCHIDS'].append(Order('ORCHIDS', best_sell['ORCHIDS'], vol)) 
        return orders



    def calc_next_price_STARFRUIT(self):
        coef = [0.39374153, 0.32139952, 0.28181973]
        intercept = 15.361666971302839
        nxt_price = intercept
        for i, val in enumerate(self.STARFRUIT_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    def calc_next_price_orchids(self, sunlight, humidity):
        # -0.0024122, -0.12234782, 
        # coef = [0.81719917, 0.09297269, 0.07895589]
        coef = [-0.04993152 -2.9182827 ] #muhammad added
        intercept = 1408.156894273624
        nxt_price = intercept
        # + (coef[0] * sunlight) + (coef[1] * humidity)
        for i, val in enumerate(self.STARFRUIT_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                # for selling
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val

    def compute_orders_AMETHYSTS(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product] # current position

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product] < 0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr) / 2
        mprice_ours = (acc_bid + acc_ask) / 2

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid - 1) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask + 1)

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < 0):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy + 1, acc_bid - 1), num))
            cpos += num

        if (cpos < self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 15):
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, min(undercut_buy - 1, acc_bid - 1), num))
            cpos += num

        if cpos < self.POSITION_LIMIT['AMETHYSTS']:
            num = min(40, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid > acc_ask) or ((self.position[product]>0) and (bid == acc_ask))) and cpos > -self.POSITION_LIMIT['AMETHYSTS']:
                order_for = max(-vol, -self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] > 0):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell-1, acc_ask+1), num))
            cpos += num

        if (cpos > -self.POSITION_LIMIT['AMETHYSTS']) and (self.position[product] < -15):
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, max(undercut_sell+1, acc_ask+1), num))
            cpos += num

        if cpos > -self.POSITION_LIMIT['AMETHYSTS']:
            num = max(-40, -self.POSITION_LIMIT['AMETHYSTS']-cpos)
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders


    def compute_orders_starfruits(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product]>0) and (bid+1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT-cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT-cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    

    def compute_orders_orchids(self, observations, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        # sort sell orders in ascending order of price (lowest first)
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        # sort buy orders in descending order of price (highest first)
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        # get south island information

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid) # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num
        
        cpos = self.position[product]
        

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert(order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders
    

    def compute_orders(self, product, order_depth, acc_bid, acc_ask, observations):

        if product == "AMETHYSTS":
            return self.compute_orders_AMETHYSTS(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_starfruits(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        if product == "ORCHIDS":
            return self.compute_orders_orchids(observations, product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        print("Observation:", state.observations)

        # Initialize the method output dict as an empty dict

        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : []}

        # set variables (will change by iterations)

        timestamp = state.timestamp
        INF = 1e9
        observations = state.observations

        # Iterate over all the keys (the available products) contained in the order depths

        for key, val in state.position.items():
            self.position[key] = val

        # setting amethysts price bounds

        AMETHYSTS_lb = 10000
        AMETHYSTS_ub = 10000

        # parsing starfruit data from OrderDepth

        if len(self.STARFRUIT_cache) == self.STARFRUIT_dim:
            self.STARFRUIT_cache.pop(0)

        _, bs_STARFRUIT = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_STARFRUIT = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        self.STARFRUIT_cache.append((bs_STARFRUIT + bb_STARFRUIT) / 2)

        STARFRUIT_lb = -INF
        STARFRUIT_ub = INF

        if len(self.STARFRUIT_cache) == self.STARFRUIT_dim:
            STARFRUIT_lb = self.calc_next_price_STARFRUIT() - 1
            STARFRUIT_ub = self.calc_next_price_STARFRUIT() + 1

        # parsing orchids data from OrderDepth and TradingState

        orchids_lb = -INF
        orchids_ub = -INF

        

        # price bounds for all products

        # we want to buy at slightly below
        acc_bid = {'AMETHYSTS' : AMETHYSTS_lb, 'STARFRUIT' : STARFRUIT_lb, 'ORCHIDS' : orchids_lb}
        # we want to sell at slightly above
        acc_ask = {'AMETHYSTS' : AMETHYSTS_ub, 'STARFRUIT' : STARFRUIT_ub, 'ORCHIDS' : orchids_ub}

        self.steps += 1

        # compute orders for all products

        for product in ['AMETHYSTS', 'STARFRUIT']: #, 'ORCHIDS':
            order_depth: OrderDepth = state.order_depths[product]
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], observations)
            result[product] += orders

        # Send final orders

        print("Result: ", result)
        print()
        # String value holding TraderState data required. It will be delivered as TradingState.traderData on next execution.
        traderData = "NameError"
        conversions = 0

        return result, conversions, traderData