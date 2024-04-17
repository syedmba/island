from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

empty_dict = {'AMETHYSTS' : 0, 'STARFRUIT' : 0, 'ORCHIDS' : 0, 'CHOCOLATE' : 0, 'STRAWBERRIES': 0, 'ROSE' : 0, 'GIFT_BASKET' : 0}


def def_value():
    return copy.deepcopy(empty_dict)

INF = int(1e9)

class Trader:

    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'AMETHYSTS' : 20, 'STARFRUIT' : 20, 'ORCHIDS' : 100, 'CHOCOLATE' : 250, 'STRAWBERRIES': 350, 'ROSE' : 60, 'GIFT_BASKET' : 60}
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

    HUMIDITY_cache = []
    HUMIDITY_dim = 50

    SUNLIGHT_cache = []
    SUNLIGHT_dim = 50

    steps = 0

    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0
    std = 25
    basket_std = 117
    
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

    sunlight_until_now = 0
    total_timestamps_hour = 10000/12 # timestamps for an hour on the SOUTH ISLAND !

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

    def calc_next_SUNLIGHT(self):
        coef = [-0.00431689]
        intercept = 2115.860341590197
        # Accuracy: 0.9977412465378546    
        nxt_price = intercept
        for i, val in enumerate(self.SUNLIGHT_cache):
            nxt_price += val * coef[i]
        return int(round(nxt_price))
    
    def calc_next_HUMIDITY(self):
        # Coefficients: [0.00013844]
        # Intercept: 70.91250876799788
        # Accuracy: 0.9957378372532794

        coef =[ 50092.47227086169, 23849.29583416846, 19645.33142050289, 24406.713853054385, 
                       13915.190301035656, 11653.674953643347, 7408.9130439089195, -1961.614822387029, 
                       24586.551143033274, -24159.158032374235, -31340.444185253902, -18483.14792592816, 
                       -30784.44757694956, -37787.72064189821, -42660.535363527815, -41525.13151780734, 
                       -32836.73988590403, -4964.965588924674, -30248.11409240085, -27734.712517593453, 
                       7084.070523296734, 4154.674737321573, -17453.928465342146, -12001.899681643261, 
                       14075.199761027403, 23769.715602819007, 4392.68528325567, 4195.87595367201, 
                       17853.580586032615, 1419.585198485245, 15045.646568365455, 27423.617789224198, 
                       -9937.545202194678, 1176.4453009003478, 26684.3245392882, 33707.62692608268, 
                       14357.096683454778, 28936.978230880453, 29278.96932031025, 38652.09257593764, 
                       6328.939769602227, -11030.578747555735, -17322.717238482364, -11174.26820224621, 
                       -28633.066086474275, -35578.39589674345, 4019.4242714812835, -4614.166458754578, 
                       -3404.0378351370564, -2577.330696944327 ]
        intercept = 8700.776027680733
        # Accuracy: 0.6052260789312435

        # coef = [0.00013844]
        # intercept = 70.91250876799788
          
        nxt_price = intercept
        for i, val in enumerate(self.HUMIDITY_cache):
            nxt_price += val * coef[i]
        return int(round(nxt_price))

    

    def calc_next_price_STARFRUIT(self):
        coef = [0.39374153, 0.32139952, 0.28181973]
        intercept = 15.361666971302839
        nxt_price = intercept
        for i, val in enumerate(self.STARFRUIT_cache):
            nxt_price += val * coef[i]

        return int(round(nxt_price))

    def calc_next_price_orchids(self, sunlight, humidity):
        hours_of_sunlight_until_now = 0 # need to increment this using sunlight values until present time
        # then we predict sunlight later on and see if 7 hours benchmark will be surpassed or not

        # for humidity it is straightforward
        # predicting next humidity can help predict quickly incoming price changes
        next_sunlight = self.calc_next_SUNLIGHT()
        next_humidity = self.calc_next_HUMIDITY()
        # if next_sunlight <= 

        # if next_humidity <

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

        # sort sell orders in ascending order of price (lowest first)
        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        # sort buy orders in descending order of price (highest first)
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        # if buying from south island is better than trading on exchange
        if south_buy_cost < best_buy_pr:
            print("buying from south")
            with_south = True
            cpos = self.position[product]
            if cpos < LIMIT:
                vol = LIMIT - cpos
                cpos += vol
                assert(vol >= 0)
                orders.append(Order(product, south_ask, vol))
        # if selling to south island is better than trading on exchange
        if south_sell_value > best_sell_pr:
            print("selling to south")
            with_south = True
            cpos = self.position[product]
            if cpos > -LIMIT:
                vol = -LIMIT - cpos
                cpos += vol
                assert(vol <= 0)
                orders.append(Order(product, south_bid, vol))

        if with_south is False:
            print("trading with self exchange")
            cpos = self.position[product]

            # to find buying opportunity in own exchange    
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
            
            # to find selling opportunity in own exchange
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

        return with_south, orders
    
    def compute_orders_basket(self, order_depth):
        
        orders = {'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSE' : [], 'GIFT_BASKET' : []}
        prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSE', 'GIFT_BASKET']
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
                if vol_buy[p] >= self.POSITION_LIMIT[p]/10:
                    break
            for price, vol in osell[p].items():
                vol_sell[p] += -vol 
                if vol_sell[p] >= self.POSITION_LIMIT[p]/10:
                    break

        res_buy = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSE'] - 375
        res_sell = mid_price['GIFT_BASKET'] - mid_price['CHOCOLATE']*4 - mid_price['STRAWBERRIES']*6 - mid_price['ROSE'] - 375

        trade_at = self.basket_std*0.5
        close_at = self.basket_std*(-1000)

        pb_pos = self.position['GIFT_BASKET']
        pb_neg = self.position['GIFT_BASKET']

        uku_pos = self.position['ROSE']
        uku_neg = self.position['ROSE']


        basket_buy_sig = 0
        basket_sell_sig = 0

        if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_buy_basket_unfill = 0
        if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
            self.cont_sell_basket_unfill = 0

        do_bask = 0

        if res_sell > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0 # no need to buy rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_sell_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol)) 
                self.cont_sell_basket_unfill += 2
                pb_neg -= vol
                #uku_pos += vol
        elif res_buy < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0 # no need to sell rn
            assert(vol >= 0)
            if vol > 0:
                do_bask = 1
                basket_buy_sig = 1
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                self.cont_buy_basket_unfill += 2
                pb_pos += vol

        if int(round(self.person_position['Olivia']['ROSE'])) > 0:

            val_ord = self.POSITION_LIMIT['ROSE'] - uku_pos
            if val_ord > 0:
                orders['ROSE'].append(Order('ROSE', worst_sell['ROSE'], val_ord))
        if int(round(self.person_position['Olivia']['ROSE'])) < 0:

            val_ord = -(self.POSITION_LIMIT['ROSE'] + uku_neg)
            if val_ord < 0:
                orders['ROSE'].append(Order('ROSE', worst_buy['ROSE'], val_ord))

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask, observations):

        if product == "AMETHYSTS":
            return self.compute_orders_AMETHYSTS(product, order_depth, acc_bid, acc_ask)
        if product == "STARFRUIT":
            return self.compute_orders_starfruits(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        if product == "ORCHIDS":
            return self.compute_orders_orchids(observations, product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
        
        
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Initialize the method output dict as an empty dict

        result = {'AMETHYSTS' : [], 'STARFRUIT' : [], 'ORCHIDS' : [], 'CHOCOLATE' : [], 'STRAWBERRIES': [], 'ROSE' : [], 'GIFT_BASKET' : []}

        # set variables (will change by iterations)

        timestamp = state.timestamp
        INF = 1e9
        observations = state.observations
        depths = state.order_depths
        conversions = 0

        if timestamp <= 100:
            self.sunlight_until_now = 0
        self.sunlight_until_now += observations.conversionObservations["ORCHIDS"].sunlight * (1 / self.total_timestamps_hour)


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

        # parsing orchids data from OrderDepth and TradingState to implement VWAP

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
        print(f"orchids buy price: {orchids_lb} for order_book: {orchid_buys}")

        for price, vol in orchid_sells.items():
            s += (price * vol)
            s_vol += vol

        orchids_ub = s / s_vol
        print(f"orchids sell price: {orchids_ub} for order_book: {orchid_sells}")

        # price bounds for all products

        # we want to buy at slightly below
        acc_bid = {'AMETHYSTS' : AMETHYSTS_lb, 'STARFRUIT' : STARFRUIT_lb, 'ORCHIDS' : orchids_lb}
        # we want to sell at slightly above
        acc_ask = {'AMETHYSTS' : AMETHYSTS_ub, 'STARFRUIT' : STARFRUIT_ub, 'ORCHIDS' : orchids_ub}

        self.steps += 1

        orders = self.compute_orders_basket(state.order_depths)
        result['GIFT_BASKET'] += orders['GIFT_BASKET']
        result['CHOCOLATE'] += orders['CHOCOLATE']
        result['STRAWBERRIES'] += orders['STRAWBERRIES']
        result['ROSE'] += orders['ROSE']

        # compute orders for all products

        for product in ['AMETHYSTS', 'STARFRUIT', 'ORCHIDS']:
            order_depth: OrderDepth = state.order_depths[product]
            if product == "ORCHIDS":
                south, orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], observations)
                if south is True:
                    conversions += 1
            else:
                orders = self.compute_orders(product, order_depth, acc_bid[product], acc_ask[product], observations)
            result[product] += orders

        # Send final orders

        print("Result: ", result)
        print()
        # String value holding TraderState data required. It will be delivered as TradingState.traderData on next execution.
        traderData = "NameError"

        return result, conversions, traderData