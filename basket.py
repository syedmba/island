import numpy as np
import pandas as pd
import json
from datamodel import Order, OrderDepth, ProsperityEncoder, TradingState, Symbol 
from typing import Any, Dict, List

#################################################################################

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]]) -> None:
        logs = self.logs
        if logs.endswith("\n"):
            logs = logs[:-1]

        print(json.dumps({
            "state": state,
            "orders": orders,
            "logs": logs,
        }, cls=ProsperityEncoder, separators=(",", ":"), sort_keys=True))

        self.state = None
        self.orders = {}
        self.logs = ""

logger = Logger()

##################################################################################

class Trader:
    def __init__(self):
        self.basket_prev = None
        self.chocolate_prev = None
        self.strawberries_prev = None
        self.roses_prev = None
        self.etf_returns = np.array([])
        self.asset_returns = np.array([])

    def run(self, state: TradingState) -> Dict[Symbol, List[Order]]:
        result = {}
        
        for product in state.order_depths.keys():
            orders_gift_basket: list[Order] = []
            orders_chocolate: list[Order] = []
            orders_strawberries: list[Order] = []
            orders_roses: list[Order] = []

            if product == 'GIFT_BASKET':
                # current positions
                basket_pos = state.position.get("GIFT_BASKET", 0)
                chocolate_pos = state.position.get("CHOCOLATE", 0)
                strawberries_pos = state.position.get("STRAWBERRIES", 0)
                roses_pos = state.position.get("ROSES", 0)

##################################################################################
                
                basket_buy_orders: Dict[int, int] = state.order_depths[product].buy_orders
                basket_sell_orders: Dict[int, int] = state.order_depths[product].sell_orders

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

                roses_price: float = (roses_best_bid + roses_best_ask) / 2

                est_price: float = 6 * strawberries_price + 4 * chocolate_price + roses_price

                price_nav_ratio: float = basket_price / est_price

##################################################################################

                self.etf_returns = np.append(self.etf_returns, basket_price)
                self.asset_returns = np.append(self.asset_returns, est_price)

                rolling_mean_etf = np.mean(self.etf_returns[-10:])
                rolling_std_etf = np.std(self.etf_returns[-10:])

                rolling_mean_asset = np.mean(self.asset_returns[-10:])
                rolling_std_asset = np.std(self.asset_returns[-10:])

                z_score_etf = (self.etf_returns[-1] - rolling_mean_etf) / rolling_std_etf
                z_score_asset = (self.asset_returns[-1] - rolling_mean_asset) / rolling_std_asset

                z_score_diff = z_score_etf - z_score_asset

                print(f'ZSCORE DIFF = {z_score_diff}')

                # implement stop loss
                # stop_loss = 0.01

                #if price_nav_ratio < self.basket_pnav_ratio - self.basket_eps:
                if z_score_diff < -2.2:
                    # stop_loss_price = self.etf_returns[-2] 


                    # ETF is undervalued! -> we buy ETF and sell individual assets!
                    # Finds volume to buy that is within position limit
                    #basket_best_ask_vol = max(basket_pos-self.basket_limit, state.order_depths['GIFT_BASKET'].sell_orders[basket_best_ask])
                    basket_best_ask_vol = state.order_depths['GIFT_BASKET'].sell_orders[basket_best_ask]
                    chocolate_best_bid_vol =  state.order_depths['CHOCOLATE'].buy_orders[chocolate_best_bid]
                    strawberries_best_bid_vol = state.order_depths['STRAWBERRIES'].buy_orders[strawberries_best_bid]
                    roses_best_bid_vol = state.order_depths['ROSES'].buy_orders[roses_best_bid]

                    # print("#"*100)
                    # print(basket_best_ask_vol, chocolate_best_bid_vol, strawberries_best_bid_vol, roses_best_bid_vol)

                    limit_mult = min(-basket_best_ask_vol, roses_best_bid_vol, 
                                     round(chocolate_best_bid_vol / 4), round(strawberries_best_bid_vol / 6))

                    print(f'LIMIT: {limit_mult}')

                    print("BUY", 'GIFT_BASKET', limit_mult, "x", basket_best_ask)
                    orders_gift_basket.append(Order('GIFT_BASKET', basket_best_ask, limit_mult))
                    
                    """
                    #chocolate_best_bid_vol = min(self.chocolate_limit-chocolate_pos, state.order_depths['CHOCOLATE'].buy_orders[chocolate_best_bid])
                    print("SELL", "CHOCOLATE", 4 * limit_mult, "x", chocolate_best_bid)
                    orders_chocolate.append(Order("CHOCOLATE", chocolate_best_bid, -4 * limit_mult))
                    
                    #strawberries_best_bid_vol = min(self.strawberries_limit-strawberries_pos, state.order_depths['STRAWBERRIES'].buy_orders[strawberries_best_bid])
                    print("SELL", "STRAWBERRIES", 6 * limit_mult, "x", strawberries_best_bid)
                    orders_strawberries.append(Order("STRAWBERRIES", strawberries_best_bid, -6 * limit_mult))
                    
                    #roses_best_bid_vol = min(self.roses_limit-roses_pos, state.order_depths['ROSES'].buy_orders[roses_best_bid])
                    print("SELL", "ROSES", limit_mult, "x", roses_best_bid)
                    orders_roses.append(Order("ROSES", roses_best_bid, -limit_mult))
                    """
                    
                     
                #elif price_nav_ratio > self.basket_pnav_ratio + self.basket_eps:
                elif z_score_diff > 2.2:
                    # ETF is overvalued! -> we sell ETF and buy individual assets!
                    # Finds volume to buy that is within position limit
                    #basket_best_bid_vol = min(self.basket_limit-basket_pos, state.order_depths['GIFT_BASKET'].buy_orders[basket_best_bid])
                    basket_best_bid_vol = state.order_depths['GIFT_BASKET'].buy_orders[basket_best_bid]
                    chocolate_best_ask_vol = state.order_depths['CHOCOLATE'].sell_orders[chocolate_best_ask]
                    strawberries_best_ask_vol = state.order_depths['STRAWBERRIES'].sell_orders[strawberries_best_ask]
                    roses_best_ask_vol = state.order_depths['ROSES'].sell_orders[roses_best_ask]

                    # print("#"*100)
                    # print(basket_best_bid_vol, chocolate_best_ask_vol, strawberries_best_ask_vol, roses_best_ask_vol)

                    limit_mult = min(basket_best_bid_vol, -roses_best_ask_vol, 
                                     round(-chocolate_best_ask_vol / 4), round(-strawberries_best_ask_vol / 6))

                    print(f'LIMIT: {limit_mult}')

                    print("SELL", 'GIFT_BASKET', limit_mult, "x", basket_best_bid)
                    orders_gift_basket.append(Order('GIFT_BASKET', basket_best_bid, -limit_mult))
                    
                    #chocolate_best_ask_vol = max(chocolate_pos-self.chocolate_limit, state.order_depths['CHOCOLATE'].sell_orders[chocolate_best_ask])
                    print("BUY", "CHOCOLATE", 4 * limit_mult, "x", chocolate_best_ask)
                    orders_chocolate.append(Order("CHOCOLATE", chocolate_best_ask, 4 * limit_mult)) 
                    
                    #strawberries_best_ask_vol = max(strawberries_pos-self.strawberries_limit, state.order_depths['STRAWBERRIES'].sell_orders[strawberries_best_ask])
                    print("BUY", "STRAWBERRIES", 6 * limit_mult, "x", strawberries_best_ask)
                    orders_strawberries.append(Order("STRAWBERRIES", strawberries_best_ask, 6 * limit_mult))
                    
                    #roses_best_ask_vol = max(roses_pos-self.roses_limit, state.order_depths['ROSES'].sell_orders[roses_best_ask])
                    print("BUY", "ROSES", limit_mult, "x", roses_best_ask)
                    orders_roses.append(Order("ROSES", roses_best_ask, limit_mult))
                    

                result['GIFT_BASKET'] = orders_gift_basket
                result['CHOCOLATE'] = orders_chocolate
                result['STRAWBERRIES'] = orders_strawberries
                result['ROSES'] = orders_roses
        
        logger.flush(state, orders_gift_basket+orders_chocolate+orders_strawberries+orders_roses) # type: ignore
        return result, 0, "NameError"