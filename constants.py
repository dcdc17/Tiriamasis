indices_path = "indices"
commodities_path = "commodities"
tickers = ["^GSPC", "^DJI", "^IXIC", "^FTSE", "^GDAXI", "^FCHI", "^N100", "EURUSD=X", "^HSI", "^DXS", "GD=F",
           "EURRUB=X"]
metals = ["GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "ALI=F"]

metal_pairs = {"GC=F": ["EURRUB=X", "^HSI", "^DXS"],
               "SI=F": ["^IXIC", "GD=F", "^GSPC", "EURRUB=X"],
               "PL=F": ["EURUSD=X", "GD=F"],
               "PA=F": ["^HSI", "^DXS"],
               "HG=F": ["^IXIC", "GD=F", "^GSPC", "^DJI"],
               "ALI=F": ["EURRUB=X", "GD=F", "^GDAXI", "^DJI", "^GSPC"]}

# index: (p, d, q)
ts_order = {'^GSPC': (2, 1, 2),
            '^DJI': (3, 1, 3),
            '^IXIC': (2, 1, 2),
            '^FTSE': (0, 1, 0),
            '^GDAXI': (2, 1, 2),
            '^FCHI': (0, 1, 0),
            '^N100': (0, 1, 0),
            'EURUSD=X': (0, 1, 0),
            '^HSI': (0, 1, 0),
            '^DXS': (2, 1, 4),
            'GD=F': (2, 1, 1),
            'EURRUB=X': (1, 1, 3),
            'GC=F': (3, 1, 3),
            'SI=F': (2, 1, 0),
            'PL=F': (0, 1, 0),
            'PA=F': (2, 1, 1),
            'HG=F': (0, 1, 0),
            'ALI=F': (0, 1, 1)}
