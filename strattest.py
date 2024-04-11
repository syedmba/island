from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np
import pandas as pd
import jsonpickle
from typing import Dict, List
# import statistics

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0}


# price_history_starfruit = np.array([])
# price_history_amethysts = np.array([])



def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:

    def __init__(self):
        price_history_starfruit = np.array([])
        # price_history_amethysts = np.array([])

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda : 0)
    # STARFRUIT_cache = []
    
    # STARFRUIT_dim = 4
    
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


    # def calc_next_price_STARFRUIT(self):
    #     # STARFRUIT cache stores price from 1 day ago, current day resp
    #     # by price, here we mean mid price

    #     coef = [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892]
    #     intercept = 4.481696494462085
    #     nxt_price = intercept
    #     for i, val in enumerate(self.STARFRUIT_cache):
    #         nxt_price += val * coef[i]

    #     return int(round(nxt_price))

    def get_sma(self, prices, rate):
        return np.mean(prices.rolling(window=rate))

    def get_std(self, prices, rate):
        return np.std(prices.rolling(window=rate))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if(buy==0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def simple_moving_average(self, product, period): 
        # product is starfruit for now
        # need to read last few timestamps (what is a good period?) 
        # do we only buy exactly at crossover or do we check crossover that is "reasonably recent"
        

        return

    def compute_orders_AMETHYSTS(self, product, order_depth, acc_bid, acc_ask):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product] # current position

        mx_with_buy = -1

        for ask, vol in osell.items():
            if ((ask < acc_bid) or ((self.position[product]<0) and (ask == acc_bid))) and cpos < self.POSITION_LIMIT['AMETHYSTS']:
                mx_with_buy = max(mx_with_buy, ask)
                order_for = min(-vol, self.POSITION_LIMIT['AMETHYSTS'] - cpos)
                cpos += order_for
                assert(order_for >= 0)
                orders.append(Order(product, ask, order_for))

        mprice_actual = (best_sell_pr + best_buy_pr)/2
        mprice_ours = (acc_bid + acc_ask)/2

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

    
    def compute_orders_STARFRUIT(self, product, order_depth, state, price_history_starfruit):
        orders: list[Order] = []

        start_trading = 2100
        position_limit = 20
        current_position = state.position.get("STARFRUIT", 0)
        history_length = 10
        spread = 3
        
        print("price_history_starfruit:", price_history_starfruit)
        price = 0
        count = 0.000001

        for Trade in state.market_trades.get("STARFRUIT", []):
            print("Trade:", Trade)
            price += Trade.price * Trade.quantity
            count += Trade.quantity
        current_avg_market_price = price / count
        
        price_history_starfruit = np.append(price_history_starfruit, current_avg_market_price)
        if len(price_history_starfruit) >= history_length+1:
            price_history_starfruit = price_history_starfruit[1:]
        
        rate = 20
        m = 2 # of std devs
            
        if state.timestamp >= start_trading:

            df_starfruit_prices = pd.DataFrame(price_history_starfruit, columns=['mid_price'])
            
            print("----------------------")
            print(df_starfruit_prices)
            sma = self.get_sma(df_starfruit_prices['mid_price'], rate).to_numpy()
            std = self.get_std(df_starfruit_prices['mid_price'], rate).to_numpy()
            print("AaAAAAAAAAAA")
            print(sma, std)

            upper_curr = sma[-1] + m * std
            upper_prev = sma[-2] + m * std
            lower_curr = sma[-1] - m * std
            lower_prev = sma[-2] - m * std
            print(lower_prev)

            # holder = price_history_starfruit[-2]
            print(price_history_starfruit)
            print(price_history_starfruit[-2])

            # if len(order_depth.sell_orders) > 0:

            #     best_ask = min(order_depth.sell_orders.keys())
            #     best_ask_volume = order_depth.sell_orders[best_ask]
            #     print(price_history_starfruit)
            #     if price_history_starfruit[-2] > lower_prev:
            #         if best_ask <= lower_curr:
            #             if np.abs(best_ask_volume) > 0:
            #                 print("BUY", "STARFRUIT", str(-best_ask_volume) + "x", best_ask)
            #                 orders.append(Order("STARFRUIT", best_ask, -best_ask_volume))

            # if len(order_depth.buy_orders) != 0:
            #     best_bid = max(order_depth.buy_orders.keys())
            #     best_bid_volume = order_depth.buy_orders[best_bid]
                
            #     if price_history_starfruit[-2] < upper_prev:
            #         if best_bid >= upper_curr:
            #             if best_bid_volume > 0:
            #                 print("SELL", "STARFRUIT", str(best_bid_volume) + "x", best_bid)
            #                 orders.append(Order("STARFRUIT", best_bid, -best_bid_volume))

        return orders, price_history_starfruit


    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product]<0) and (ask == acc_bid+1))) and cpos < LIMIT:
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
    

    def compute_orders(self, product, order_depth, acc_bid, acc_ask, state, price_history_starfruit):

        if product == "AMETHYSTS":
            return self.compute_orders_AMETHYSTS(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_STARFRUIT(product, order_depth, state, price_history_starfruit)[0]
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {'AMETHYSTS' : [], 'STARFRUIT' : []}

        # global price_history_amethysts
        global price_history_starfruit
        
        try:
            price_history_starfruit = jsonpickle.decode(state.traderData)
        except:
            price_history_starfruit = np.array([])

        # Iterate over all the keys (the available products) contained in the order dephts
        for key, val in state.position.items():
            self.position[key] = val
        print()
        for key, val in self.position.items():
            print(f'{key} position: {val}')

        # assert abs(self.position.get('UKULELE', 0)) <= self.POSITION_LIMIT['UKULELE']

        timestamp = state.timestamp

        # if len(self.STARFRUIT_cache) == self.STARFRUIT_dim:
        #     self.STARFRUIT_cache.pop(0)

        # _, bs_STARFRUIT = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        # _, bb_STARFRUIT = self.values_extract(collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)

        # self.STARFRUIT_cache.append((bs_STARFRUIT+bb_STARFRUIT)/2)

        INF = 1e9
    
        STARFRUIT_lb = -INF
        STARFRUIT_ub = INF

        # if len(self.STARFRUIT_cache) == self.STARFRUIT_dim:
        #     STARFRUIT_lb = self.calc_next_price_STARFRUIT()-1
        #     STARFRUIT_ub = self.calc_next_price_STARFRUIT()+1

        AMETHYSTS_lb = 10000
        AMETHYSTS_ub = 10000

        # CHANGE FROM HERE

        acc_bid = {'AMETHYSTS' : AMETHYSTS_lb, 'STARFRUIT' : STARFRUIT_lb} # we want to buy at slightly below
        acc_ask = {'AMETHYSTS' : AMETHYSTS_ub, 'STARFRUIT' : STARFRUIT_ub} # we want to sell at slightly above

        self.steps += 1


        # for product in ['AMETHYSTS', 'STARFRUIT']:
        #     order_depth: OrderDepth = state.order_depths[product]
        #     orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], state)
        #     result[product] += orders

        product = "AMETHYSTS"
        order_depth: OrderDepth = state.order_depths[product]
        result[product] += self.compute_orders_AMETHYSTS(product, order_depth, acc_bid[product], acc_ask[product])

        product = "STARFRUIT"
        order_depth: OrderDepth = state.order_depths[product]
        order, price_history_starfruit = self.compute_orders_STARFRUIT(product, order_depth, state, price_history_starfruit)
        result[product] += order


        totpnl = 0

        print(f"Timestamp {timestamp}, Total PNL ended up being {totpnl}")
        # print(f'Will trade {result}')
        print("End transmission")
        print(result)
        state.traderData = jsonpickle.encode(price_history_starfruit) 
        # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
       
        traderData = state.traderData

        conversions = 0
        return result, conversions, traderData