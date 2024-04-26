from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np
import statistics

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, "ORCHIDS" : 0, 'CHOCOLATE' : 0, 'STRAWBERRIES': 0, 'ROSES' : 0, 'GIFT_BASKET' : 0, "COCONUT": 0, "COCONUT_COUPON": 0}


def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, "ORCHIDS": 100, 'CHOCOLATE' : 250, 'STRAWBERRIES': 350, 'ROSES' : 60, 'GIFT_BASKET' : 60, "COCONUT": 300, "COCONUT_COUPON": 600}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cpnl = defaultdict(lambda : 0)
    
    STARFRUIT_cache = []
    STARFRUIT_dim = 3

    sun_cache = []
    sun_dim = 3

    hum_cache = []
    hum_dim = 3

    coc_cache = []
    coc_dim = 3

    steps = 0

    # cont_buy_basket_unfill = 0
    # cont_sell_basket_unfill = 0
    std = 25
    basket_std = 76.42
    choco_std = 99.34436123456715
    straw_std = 27.405575223133585
    rose_std = 161.42099979958462
    
    halflife_diff = 5
    alpha_diff = 1 - np.exp(-np.log(2)/halflife_diff)

    halflife_price = 5
    alpha_price = 1 - np.exp(-np.log(2)/halflife_price)

    halflife_price_CHOCOLATE = 20
    alpha_price_CHOCOLATE = 1 - np.exp(-np.log(2)/halflife_price_CHOCOLATE)
    
    begin_diff_CHOCOLATE = -INF
    begin_diff_bag = -INF
    begin_bag_price = -INF
    begin_CHOCOLATE_price = -INF

    def newton_raphson():
        pass

    def normCdf(self, x):
        # pre-determined values using rational minimax approximation. No need to worry about them as they're just here to give a good estimate.
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911

        # determining sign of output
        sign = 1 if (x >= 0) else -1
        x = abs(x) / math.sqrt(2.0); # standardizing x

        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)

        return 0.5 * (1.0 + sign * y)
    
    def black_scholes_call(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * self.normCdf(d1) - K * np.exp(-r*T) * self.normCdf(d2)
    
    def black_scholes_put(self, S, K, T, r, sigma):
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r*T) * self.normCdf(-d2) - S * self.normCdf(-d1)
    
    def calc_next_price_coconut(self):
        coef = [0.96842006, 0.03749103, -0.00594672]
        intercept = 0.35228277771784633
        nxt_coc = intercept
        for i, val in enumerate(self.coc_cache):
            nxt_coc += val * coef[i]

        return int(round(nxt_coc))

    def predict_sunlight(self):
        coef = [1.03037314e+00, -2.25013886e-06, -3.03795021e-02]
        intercept = 0.017696349974357872
        nxt_sun = intercept
        for i, val in enumerate(self.sun_cache):
            nxt_sun += val * coef[i]

        return int(round(nxt_sun))
    
    def predict_humidity(self):
        coef = [1.01254311e+00, -3.38481111e-06, -1.25578231e-02]
        intercept = 0.0011386486631153048
        nxt_hum = intercept
        for i, val in enumerate(self.hum_cache):
            nxt_hum += val * coef[i]

        return int(round(nxt_hum))

    def calc_next_price_STARFRUIT(self):
        coef = [0.39374153, 0.32139952, 0.28181973]
        intercept = 15.361666971302839
        nxt_price = intercept
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
        with_south = False # whether to trade with south island or not
        # south island info
        south_island = observations.conversionObservations["ORCHIDS"]
        south_bid = south_island.bidPrice
        south_ask = south_island.askPrice
        south_transport_fees = south_island.transportFees
        south_export_tariff = south_island.exportTariff
        south_import_tariff = south_island.importTariff
        south_sunlight = south_island.sunlight
        south_humidity = south_island.humidity

        # net cost of buying from south island
        south_buy_cost = south_ask + south_transport_fees + south_import_tariff
        # net value from selling to south island
        south_sell_value = south_bid - south_transport_fees - south_export_tariff

        # adjust sunlight and humidity information, and make next predictions
        if len(self.sun_cache) == self.sun_dim:
            self.sun_cache.pop(0)

        self.sun_cache.append(south_sunlight)

        if len(self.hum_cache) == self.hum_dim:
            self.hum_cache.pop(0)

        self.hum_cache.append(south_humidity)

        next_sun = self.predict_sunlight()
        next_hum = self.predict_humidity()

        # sort sell orders in ascending order of price (lowest first)
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        # sort buy orders in descending order of price (highest first)
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        print(f"best sell price for orchids is {best_sell_pr}")
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)
        print(f"best buy price for orchids is {best_buy_pr}")

        # if sunlight or humidity out of range, then supply decreases so price increases, so long
        if next_hum < 60 or next_hum > 80:
            # go long
            if self.position[product] < LIMIT:
                vol = LIMIT - self.position[product]
                orders.append(Order("ORCHIDS", best_buy_pr, vol))
                self.position[product] += vol
        # if sunlight and humidity in range later, supply increases so price decreases, so short
        if (south_humidity < 60 or south_humidity > 80) and (next_hum > 60 and next_hum < 80):
            # go short in our exchange
            if self.position[product] > -LIMIT:
                vol = -LIMIT - self.position[product]
                orders.append(Order("ORCHIDS", best_sell_pr, vol))
                self.position[product] += vol

        return with_south, orders
    

    def compute_orders_basket(self, order_depth, state):
        
        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        basket_buy_orders: Dict[int, int] = state.order_depths["GIFT_BASKET"].buy_orders
        basket_sell_orders: Dict[int, int] = state.order_depths["GIFT_BASKET"].sell_orders

        basket_best_bid: float = max(basket_buy_orders)
        basket_best_ask: float = min(basket_sell_orders)

        # Finding price / NAV ratio
        basket_price: float = (basket_best_bid + basket_best_ask) / 2

        chocolate_buy_orders: Dict[int, int] = state.order_depths['CHOCOLATE'].buy_orders
        chocolate_sell_orders: Dict[int, int] = state.order_depths['CHOCOLATE'].sell_orders

        chocolate_best_bid: float = max(chocolate_buy_orders)
        chocolate_best_ask: float = min(chocolate_sell_orders)

        chocolate_price: float = (chocolate_best_bid + chocolate_best_ask) / 2

        strawberries_buy_orders: Dict[int, int] = state.order_depths['STRAWBERRIES'].buy_orders
        strawberries_sell_orders: Dict[int, int] = state.order_depths['STRAWBERRIES'].sell_orders

        strawberries_best_bid: float = max(strawberries_buy_orders)
        strawberries_best_ask: float = min(strawberries_sell_orders)

        strawberries_price: float = (strawberries_best_bid + strawberries_best_ask) / 2

        roses_buy_orders: Dict[int, int] = state.order_depths['ROSES'].buy_orders
        roses_sell_orders: Dict[int, int] = state.order_depths['ROSES'].sell_orders

        roses_best_bid: float = max(roses_buy_orders)
        roses_best_ask: float = min(roses_sell_orders)

        roses_best_bid_vol = state.order_depths['ROSES'].buy_orders[roses_best_bid]

        roses_price: float = (roses_best_bid + roses_best_ask) / 2

        est_price: float = 6 * strawberries_price + 4 * chocolate_price + roses_price

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
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 379
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSES'] - 379

        straw_buy_sell = mid_price['STRAWBERRIES'] - 4026.83735
        choco_buy_sell = mid_price['CHOCOLATE'] - 7915.34725
        rose_buy_sell = mid_price['ROSES'] - 14506.89705

        trade_at = self.basket_std*0.5 #300
        close_at = self.basket_std*(-1000)

        gb_pos = self.position['GIFT_BASKET']
        gb_neg = self.position['GIFT_BASKET']

        rose_vol = min(self.position['ROSES'] + self.POSITION_LIMIT['ROSES'], roses_best_bid_vol)
        # print(f"ROSES ARE {self.position['ROSES']} RN  HAHAHAHHAA")
        if self.position['ROSES'] > -self.POSITION_LIMIT['ROSES']:
            orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -rose_vol))

        if res_sell > trade_at: # SELLING
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            # self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                # do_bask = 1
                # basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                # self.cont_sell_basket_unfill += 2
                gb_neg -= vol
                #uku_pos += vol

############################### VOL MAXXING TACTIC ################################

                # if self.position['STRAWBERRIES'] <= 0: #buying components
                #     straw_vol = min(-self.position['STRAWBERRIES'] + self.POSITION_LIMIT['STRAWBERRIES'], vol * 6)
                #     orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], straw_vol))
                # else:
                #     straw_vol = min(-self.position['STRAWBERRIES'] + self.POSITION_LIMIT['STRAWBERRIES'], vol * 6)
                #     orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], straw_vol))

                # if self.position['CHOCOLATE'] <= 0:
                #     choco_vol = min(-self.position['CHOCOLATE'] + self.POSITION_LIMIT['CHOCOLATE'], vol * 4)
                #     orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], choco_vol))
                # else:
                #     choco_vol = min(-self.position['CHOCOLATE'] + self.POSITION_LIMIT['CHOCOLATE'], vol * 4)
                #     orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], choco_vol))
                
                # if self.position['ROSES'] <= 0:
                #     rose_vol = min(-self.position['ROSES'] + self.POSITION_LIMIT['ROSES'], vol)
                #     orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], rose_vol))
                # else:
                #     rose_vol = min(-self.position['ROSES'] + self.POSITION_LIMIT['ROSES'], vol)
                #     orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], rose_vol))


############################### MIN VOL TACTIC ################################

            # #basket_best_bid_vol = min(self.basket_limit-basket_pos, state.order_depths['GIFT_BASKET'].buy_orders[basket_best_bid])
            # basket_best_bid_vol = state.order_depths['GIFT_BASKET'].buy_orders[basket_best_bid]
            # chocolate_best_ask_vol = state.order_depths['CHOCOLATE'].sell_orders[chocolate_best_ask]
            # strawberries_best_ask_vol = state.order_depths['STRAWBERRIES'].sell_orders[strawberries_best_ask]
            # roses_best_ask_vol = state.order_depths['ROSES'].sell_orders[roses_best_ask]

            # # print("#"*100)
            # # print(basket_best_bid_vol, chocolate_best_ask_vol, strawberries_best_ask_vol, roses_best_ask_vol)

            # limit_mult = min(basket_best_bid_vol, -roses_best_ask_vol, 
            #                     round(-chocolate_best_ask_vol / 4), round(-strawberries_best_ask_vol / 6))

            # print(f'LIMIT: {limit_mult}')

            # print("SELL", 'GIFT_BASKET', limit_mult, "x", basket_best_bid)
            # orders["GIFT_BASKET"].append(Order('GIFT_BASKET', basket_best_bid, -limit_mult))
            
            # #chocolate_best_ask_vol = max(chocolate_pos-self.chocolate_limit, state.order_depths['CHOCOLATE'].sell_orders[chocolate_best_ask])
            # print("BUY", "CHOCOLATE", 4 * limit_mult, "x", chocolate_best_ask)
            # orders["CHOCOLATE"].append(Order("CHOCOLATE", chocolate_best_ask, 4 * limit_mult)) 
            
            # #strawberries_best_ask_vol = max(strawberries_pos-self.strawberries_limit, state.order_depths['STRAWBERRIES'].sell_orders[strawberries_best_ask])
            # print("BUY", "STRAWBERRIES", 6 * limit_mult, "x", strawberries_best_ask)
            # orders["STRAWBERRIES"].append(Order("STRAWBERRIES", strawberries_best_ask, 6 * limit_mult))
            
            # #roses_best_ask_vol = max(roses_pos-self.roses_limit, state.order_depths['ROSES'].sell_orders[roses_best_ask])
            # print("BUY", "ROSES", limit_mult, "x", roses_best_ask)
            # orders["ROSES"].append(Order("ROSES", roses_best_ask, limit_mult))

        elif res_buy < -trade_at: # BUYING
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            # self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                # do_bask = 1
                # basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                # self.cont_buy_basket_unfill += 2

                gb_pos += vol

############################### MIN VOL TACTIC ################################

                # if self.position['STRAWBERRIES'] <= 0: #selling components
                #     straw_vol = min(self.position['STRAWBERRIES'] + self.POSITION_LIMIT['STRAWBERRIES'], vol * 6)
                #     orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -straw_vol))
                # else:
                #     straw_vol = min(self.position['STRAWBERRIES'] + self.POSITION_LIMIT['STRAWBERRIES'], vol * 6)
                #     orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -straw_vol))

                # if self.position['CHOCOLATE'] <= 0:
                #     choco_vol = min(self.position['CHOCOLATE'] + self.POSITION_LIMIT['CHOCOLATE'], vol * 4)
                #     orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -choco_vol))
                # else:
                #     choco_vol = min(self.position['CHOCOLATE'] + self.POSITION_LIMIT['CHOCOLATE'], vol * 4)
                #     orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -choco_vol))
                
                # if self.position['ROSES'] <= 0:
                #     rose_vol = min(self.position['ROSES'] + self.POSITION_LIMIT['ROSES'], vol)
                #     orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -rose_vol))
                # else:
                #     rose_vol = min(self.position['ROSES'] + self.POSITION_LIMIT['ROSES'], vol)
                #     orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -rose_vol))

############################### VOL MAXXING TACTIC ################################
            
            # chocolate_best_ask_vol = state.order_depths['CHOCOLATE'].sell_orders[chocolate_best_ask]
            # strawberries_best_ask_vol = state.order_depths['STRAWBERRIES'].sell_orders[strawberries_best_ask]
            # roses_best_ask_vol = state.order_depths['ROSES'].sell_orders[roses_best_ask]

            # # print("#"*100)
            # # print(basket_best_bid_vol, chocolate_best_ask_vol, strawberries_best_ask_vol, roses_best_ask_vol)

            # limit_mult = min(basket_best_bid_vol, -roses_best_ask_vol, 
            #                     round(-chocolate_best_ask_vol / 4), round(-strawberries_best_ask_vol / 6))

            # print(f'LIMIT: {limit_mult}')

            # print("SELL", 'GIFT_BASKET', limit_mult, "x", basket_best_bid)
            # orders["GIFT_BASKET"].append(Order('GIFT_BASKET', basket_best_bid, -limit_mult))
            
            # #chocolate_best_ask_vol = max(chocolate_pos-self.chocolate_limit, state.order_depths['CHOCOLATE'].sell_orders[chocolate_best_ask])
            # print("BUY", "CHOCOLATE", 4 * limit_mult, "x", chocolate_best_ask)
            # orders["CHOCOLATE"].append(Order("CHOCOLATE", chocolate_best_ask, 4 * limit_mult)) 
            
            # #strawberries_best_ask_vol = max(strawberries_pos-self.strawberries_limit, state.order_depths['STRAWBERRIES'].sell_orders[strawberries_best_ask])
            # print("BUY", "STRAWBERRIES", 6 * limit_mult, "x", strawberries_best_ask)
            # orders["STRAWBERRIES"].append(Order("STRAWBERRIES", strawberries_best_ask, 6 * limit_mult))
            
            # #roses_best_ask_vol = max(roses_pos-self.roses_limit, state.order_depths['ROSES'].sell_orders[roses_best_ask])
            # print("BUY", "ROSES", limit_mult, "x", roses_best_ask)
            # orders["ROSES"].append(Order("ROSES", roses_best_ask, limit_mult))
        

############################### PER COMPONENT OVERBOUGHT OVERSOLD TACTIC AS ACTUALLY IMPLEMENTED FOR BASKETS ################################


        # # FOR CHOCOLATE
        # if choco_buy_sell > self.choco_std*0.5: # SELLING because overvalued
        #     vol = self.position['CHOCOLATE'] + self.POSITION_LIMIT['CHOCOLATE']
        #     assert(vol >= 0)
        #     if vol > 0:
        #         orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_buy['CHOCOLATE'], -vol)) 

        # elif choco_buy_sell < -self.choco_std*0.5: # BUYING
        #     vol = self.POSITION_LIMIT['CHOCOLATE'] - self.position['CHOCOLATE']
        #     assert(vol >= 0)
        #     if vol > 0:
        #         orders['CHOCOLATE'].append(Order('CHOCOLATE', worst_sell['CHOCOLATE'], vol))
        
        # # FOR STRAWBERRIES
        # if straw_buy_sell > self.straw_std*0.5: # SELLING because overvalued
        #     vol = self.position['STRAWBERRIES'] + self.POSITION_LIMIT['STRAWBERRIES']
        #     assert(vol >= 0)
        #     if vol > 0:
        #         orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_buy['STRAWBERRIES'], -vol)) 

        # elif straw_buy_sell < -self.straw_std*0.5: # BUYING
        #     vol = self.POSITION_LIMIT['STRAWBERRIES'] - self.position['STRAWBERRIES']
        #     assert(vol >= 0)
        #     if vol > 0:
        #         orders['STRAWBERRIES'].append(Order('STRAWBERRIES', worst_sell['STRAWBERRIES'], vol))
        
        # # FOR ROSES
        # if rose_buy_sell > self.rose_std*0.5: # SELLING because overvalued
        #     vol = self.position['ROSES'] + self.POSITION_LIMIT['ROSES']
        #     assert(vol >= 0)
        #     if vol > 0:
        #         orders['ROSES'].append(Order('ROSES', worst_buy['ROSES'], -vol)) 

        # elif rose_buy_sell < -self.rose_std*0.5: # BUYING
        #     vol = self.POSITION_LIMIT['ROSES'] self.position['ROSES']
        #     assert(vol >= 0)
        #     if vol > 0:
        #         orders['ROSES'].append(Order('ROSES', worst_sell['ROSES'], vol))

        return orders

    def compute_orders_coconuts(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        # lin reg model, just like starfruits
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

    def compute_orders_coupons(self, product, order_depth):
        # strategy:
        # for every timestamp, get current price of coconut (spot price)
        # we know strike price is 10000
        # time to expiry is 250 days
        # can keep risk free rate as 4% (0.04)
        # based on this, we calculate price for coupon (option)
        # if price for coupon is higher than what we paid, sell the coupon
        # if price for coupon is lower than what we paid, buy more coupons

        pass

    def compute_orders(self, product, order_depth, acc_bid, acc_ask, observations):

        if product == "AMETHYSTS":
            return self.compute_orders_AMETHYSTS(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_starfruits(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        if product == 'ORCHIDS':
            return self.compute_orders_orchids(observations, product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        
        
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict

        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS': [], 'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSES' : [], 'GIFT_BASKET' : [], 'COCONUT': [], "COCONUT_COUPON": []}

        # set variables (will change by iterations)

        timestamp = state.timestamp
        INF = 1e9
        observations = state.observations
        depths = state.order_depths
        conversions = 0

        # Iterate over all the keys (the available products) contained in the order depths

        for key, val in state.position.items():
            self.position[key] = val

        assert abs(self.position.get('ROSES', 0)) <= self.POSITION_LIMIT['ROSES']

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

        # set bid and ask limits for orchids using VWAP

        orchids_lb = -INF
        orchids_ub = INF

        orchid_buys = depths['ORCHIDS'].buy_orders # dict of price:volume
        orchid_sells = depths['ORCHIDS'].sell_orders # dict of price:volume
        # orchid_market_trades = state.market_trades['ORCHIDS'] # list of Trade objects

        b, s, b_vol, s_vol = 0, 0, 0, 0

        for price, vol in orchid_buys.items():
            b += (price * vol)
            b_vol += vol

        orchids_lb = b / b_vol

        for price, vol in orchid_sells.items():
            s += (price * vol)
            s_vol += vol

        orchids_ub = s / s_vol

        # bid and ask limits for 3 products

        # we want to buy at slightly below
        acc_bid = {'AMETHYSTS' : AMETHYSTS_lb, 'STARFRUIT' : STARFRUIT_lb, "ORCHIDS" : orchids_lb}
        # we want to sell at slightly above
        acc_ask = {'AMETHYSTS' : AMETHYSTS_ub, 'STARFRUIT' : STARFRUIT_ub, "ORCHIDS" : orchids_ub}

        with_south, orders = self.compute_orders_orchids(observations, 'ORCHIDS', state.order_depths['ORCHIDS'], orchids_lb, orchids_ub, self.POSITION_LIMIT['ORCHIDS'])

        if with_south:
            conversions += 1

        result['ORCHIDS']  += orders

        # trading logic for baskets/components

        self.steps += 1

        orders = self.compute_orders_basket(state.order_depths, state)
        result['GIFT_BASKET'] += orders['GIFT_BASKET']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['ROSES'] += orders['ROSES']

        for product in state.own_trades.keys():
            for trade in state.own_trades[product]:
                if trade.timestamp != state.timestamp-100:
                    continue
                # print(f'We are trading {product}, {trade.buyer}, {trade.seller}, {trade.quantity}, {trade.price}')
                self.volume_traded[product] += abs(trade.quantity)
                if trade.buyer == "SUBMISSION":
                    self.cpnl[product] -= trade.quantity * trade.price
                else:
                    self.cpnl[product] += trade.quantity * trade.price

        # trading logic for coconuts and coupons

        if len(self.coc_cache) == self.coc_dim:
            self.coc_cache.pop(0)

        _, bs_coc = self.values_extract(collections.OrderedDict(sorted(state.order_depths['COCONUT'].sell_orders.items())))
        _, bb_coc = self.values_extract(collections.OrderedDict(sorted(state.order_depths['COCONUT'].buy_orders.items(), reverse=True)), 1)

        self.coc_cache.append((bs_coc + bb_coc) / 2)

        coc_lb = -INF
        coc_ub = INF

        if len(self.coc_cache) == self.coc_dim:
            coc_lb = self.calc_next_price_coconut() - 1
            coc_ub = self.calc_next_price_coconut() + 1

        coc_orders = self.compute_orders_coconuts('COCONUT', state.order_depths['COCONUT'], coc_lb, coc_ub, self.POSITION_LIMIT['COCONUT'])
        coup_orders = self.compute_orders_coupons('COCONUT_COUPON', state.order_depths['COCONUT_COUPON'])
        # result['COCONUT'] += coc_orders
        # result['COCONUT_COUPON'] += coup_orders

        # compute orders for all products

        for product in ['AMETHYSTS', 'STARFRUIT']:
            order_depth: OrderDepth = state.order_depths[product]
            
            orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], observations)
            result[product] += orders

        # Send final orders

        print("Result: ", result)
        print()
        # String value holding TraderState data required. It will be delivered as TradingState.traderData on next execution.
        traderData = "NameError"

        return result, conversions, traderData
    
# 794 (796 with this)