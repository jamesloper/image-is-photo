import os

import requests

ids = [
    "9YuEbyGWGzqCuiAGP",
    "Rtw87HcQHFkiBfW5J",
    "DvK8pf3bTrsgmEkQZ",
    "iDcTvPhyCZ8b92EWm",
    "6f4EDLa26YhQAwdxm",
    "zCakg9aC2iqKgx4Zo",
    "j3EwiRBzHJJf8c2wN",
    "93koep45knwX6wQmK",
    "AyJDnY9F97K2sRr27",
    "FXLF2nnpszJdWyXRq",
    "7g5pvECjWFCo5WcvZ",
    "znXnJfRuXQd8hvFyG",
    "3tXMTuy7RkELFMjfu",
    "qBgngtkN5kPcQae2h",
    "uX5jSB56Truy9Ptpq",
    "MxoG5jgHd35qpijqL",
    "D9yXsu2Fx5cukL59e",
    "G5h7uTLihRFmteBxK",
    "KeW79macFjTQfquia",
    "Msd5PmDB6xnz2pvoA",
    "us6epwMyfLhJ39yNm",
    "ZsE6NQWZyiuvk8sTa",
    "2mcZ8koLMTnNpqNMg",
    "5JE4BMa4SK8ECHozp",
    "ntuZYKW2rLh4qDLct",
    "t3kniNfX2wLL3BpcS",
    "bFyELtwK8pvRNyQC8",
    "tbRCtQriXhTGt9837",
    "9r6tgAm9fDPRZQDAG",
    "Mr9trWpB4wL7nXh7M",
    "xD8tp4765y3g6Gpgf",
    "bWEhyKiLdLCdDt6cX",
    "XEDvetdMxEXXdQPvJ",
    "ArARSiHzW8DaT9X5E",
    "ShCvi6qPRFRK3Hjbg",
    "GXbrSw8CjrhrzmPki",
    "BmiJYQCATTeNWLq4W",
    "FCiSZGdQxABjeRTtq",
    "hzR75uzC6XPvFMFoQ",
    "smjJz3WKRrXHH8RKs",
    "agT9KcQiXsQ6AvPDv",
    "K6YsK39LFA7AGMRQf",
    "sBvNHMzoZnArbpMvq",
    "WBAMQYMmQM2LdBDTj",
    "QSgFWoAJhqeGSePGc",
    "4AQmdvYXzaSqFD2R2",
    "s6Hu9n6dQkEiuNcpS",
    "723tY8FyxyLzXF9vp",
    "DC3Be9Qx5DRm7rvq6",
    "vKbyXT7ri5WZhPxX4",
    "o4dydHH9qDSGTdegB",
    "9jFCueWFfDZnmb66r",
    "aebibDtiQDheXYwZd",
    "te75XmxMwhKsBHnqG",
    "5PpeiWoCwwtHuAhnQ",
    "xuPfiCqo8SQQeD5bG",
    "63SfhfkehyDgZx6Wv",
    "P4QiCeJe4E8bfSqq6",
    "b2R48X7ZyqCe8h6ob",
    "X5iuHiZ7sfK8RBqof",
    "s8DJMjdSTzoiTZbYi",
    "J8XYXEPdfxqmb7mEy",
    "NcnsWGANMRN7CsPHS",
    "Ks3S9RdonKWSy8MnL",
    "BacFvspSiHWWkzAr5",
    "zNS83T2LEPZEi2Jmf",
    "eoccemdvXoLrrNLBL",
    "vuHNSa4yFXsN8zZXd",
    "SjdFoDQczc525n5Sd",
    "uLSmijngGPkvn4kyb",
    "fbcbJ6wGt77THdYwr",
    "jMeecKyTd8bWCnMdg",
    "pGK6xSRaNRkogXzck",
    "g82vL7bWhq24Ls2pq",
    "BmSXSu78oMAmdjkEi",
    "pA6ivXLKexq2uxaJt",
    "5apBerupmhhpZcwhd",
    "2gLnZhZQ8sMLeMjqz",
    "ghHtkSMSB3GAKjAfh",
    "xwkjJyy5k7tGMSZ8A",
    "9Ldj2k7cfcGo5H3t5",
    "SrTy4D4mPX7zPyvG8",
    "zXoD3patWFRzNwbCt",
    "56AtNFHwprimeykmj",
    "HfDycfi9aMgaAzuDn",
    "46vH3AyzFbZdbRNro",
    "vCRsxTT89xAEAWC72",
    "rNgPT3Y99kJjD5dxM",
    "P9n9prCfYkM92rEAg",
    "HgFfK6rxgoCDGHYKY",
    "7t2mrcmDgSydBnmbb",
    "mLibH7BDqqWBRcwEF",
    "dL2NSqCXJHRbQcvxR",
    "KqybsNJpnW9kAutEq",
    "Md5BxFZwJaN7bisFt",
    "Hz8zA5feQ8bDLg3v9",
    "XdzL9h2GAKuZxRdcY",
    "PPZd9rNikhPDEnqnz",
    "TGuwyXno2sRraSubb",
    "2EKdRMJHwq7D2HknF",
    "7aMPFPedaA4jNp5eD",
    "yK4zw7GvmBBCnMJwF",
    "4nP9b42ZbyaM4g6hc",
    "mY3Sa3k2ZghzRYo73",
    "n94D6gPhrPBSkpaaH",
    "kLdQ7FbjwiAXpJXB3",
    "woi6c43gdHRBWJhjg",
    "ZfxY2YeGx2ESZn7na",
    "KbNxo6Kz7eSZLSc35",
    "WgTX8vasShQs2fXni",
    "8kBkeNj24vPiwTtjh",
    "mSDb79anzJZAKACKF",
    "jXS5uYJLB5YyjPALw",
    "qA7uBYDnDhpgabkaz",
    "M9TnDcvMpCwALNun4",
    "XYingGioewvFRkd77",
    "cJHcDakqAMH4rtp9x",
    "ZFrniwDgs9N5NJNor",
    "fhg75DoWYvQywtAt2",
    "LJuweB5Z5kGuemPrJ",
    "NXFdZyF8sF55qw6eN",
    "yRwymm5sGZhZqCNqo",
    "8iqcxB5yFTFMqXSMT",
    "aRN2yrawiupqzZ4qJ",
    "HcBMmrudyXrDv6SNJ",
    "jebeXyCKPKXMDo6Q6",
    "BWLDRs4WLoJfMQnae",
    "k5xF5oALpHAcb8Sxx",
    "dDdms2WLb9X9j7yvv",
    "ofH7Rd9E4WoPHced4",
    "ZqgjwEKPxHEER6ZJ9",
    "HrK3ZPnYi4myPMCXY",
    "HEk5DxrCYvRfs6YSZ",
    "muATvjwRaHiLMNiQ4",
    "yqusw2MWnvRPJKP9T",
    "h7d8wa3NAXkAztvKd",
    "sXgnGPvGSSYPbJER6",
    "mMPbvjT7hgsoM4q4H",
    "QfyhAZjz8YsTCwJTt",
    "CcS7cZj9EBTxKLdbc",
    "Mj6G9kgzkZizK3SWK",
    "NmTjs5TkCGxcrmQSA",
    "3eahfzJkmZuchHDjj",
    "9uj9NT78LTpmQQ2iJ",
    "otLEJavvaNwEsvYsM",
    "gquC9mBX3LyAjapvF",
    "RhcGeq33RvJYp7nzn",
    "JzeFBbQSiSL5p9Z2y",
    "WwzADSnNpoFfkQeis",
    "xmmJSZ8enQ7dFKKij",
    "kEXf585QwEfjALARQ",
    "BYGhp7PBG2QX7eZzK",
    "JibC8hY3xyACx2ydZ",
    "HmmyGX3SCjgjbhppM",
    "REyCFyBKqNqFrLYiw",
    "35ChLjWyPeFdtxsDG",
    "GwSJudRkePxoE7CTe",
    "uuEDB698Lac3Qhmhg",
    "zGGFY5LvogBmiAY5a",
    "qv6df9D9ybbJSRW2h",
    "T82WeQ9TYDpnmsiDi",
    "cEPvaop6L9Fa2fCjJ",
    "kTHxKNYjchAZp3iAq",
    "kLSRJKrcgWvQ2Kris",
    "g92i4cwtT7HxQ59wP",
    "PFfqLeceYnFLm4afK",
    "mfZ7kgb5SCyKDoprq",
    "4GwTWwkepNnB5dsP3",
    "uwkhjHxFJCdFjpWpp",
    "pGHyyxxXuWBic2KGp",
    "hsySFtW3fEwtA5ndj",
    "N8GBefTmhfNs9vMJz",
    "EMeM7CHSKbX6HGZnS",
    "L7yWopJZeQFbGeWpk",
    "DJENPM8DDkdigsoS7",
    "hduvNoFWqespY4roQ",
    "BSDs7sLykdHZCYPua",
    "3rScSdcnqRPMh955W",
    "XNzMAqt5bspZhQ2js",
    "xwzTaEAbwydzFiJL7",
    "8cX6YGisFk2MGCyMZ",
    "W3fCqzZYgDcMLmnRW",
    "SaKqb2k4YN3rD5BHr",
    "q87RCGkT6Rc7S8WM9",
    "aoQhgui6p2bEXNNcp",
    "DwMCTZYQwzMHZqCMG",
    "wX4tLJhkHF6dJeQM5",
    "McBFACEiH8gM8stE9",
    "4PFBTGmWFci4qZdJr",
    "DSaCspCCixQ6Xz745",
    "rBi3ugtBJnCKSPXj5",
    "WKkkBYgbEdLDTGnxF",
    "Eh2cdCFHv4RNBtSHB",
    "vtGoRyrnPrknTDMmL",
    "EBkHvzzTsjRYdbHyF",
    "Ndox3PW2i4PMnYtry",
    "p6evFbZS8Bivwtm7M",
    "yedskC7EodD3j3Arc",
    "FnB36pEdqojDP5SJg",
    "yHcnx38i2wnicbP36",
    "viiDCf4Wiajd48uoE",
    "rveMWu6jPMhyxoh7y",
    "NyeNAq3XQ2FmnyJF2",
    "ujmpJPkH9gTkgEB4L",
    "bkYSdZeaBzcN9ELuA",
    "Kf3aRFrJHyZYNeJk9",
    "LWDZDg8YikSoBwEXF",
    "p4neufEQQMhxu2bfE",
    "jrA6nMkKkR2t5GAGN",
    "4zKuX3gdbm3CJ9Zgm",
    "X4nWK7NZwTFRtcN2h",
    "oyjCjTNgcbxN34PPg",
    "ks7AXg7yHDjTx6PPz",
    "JEk3z4hWpWQYCaqeK",
    "xWLACcyFR3spHNQ4T",
    "LeBb9NcsqL7Ri3v8t",
    "Jjkh2nbPm6omuvCf3",
    "HddwgDENmNimSanKF",
    "F6FtM3C3veRTPNu5c",
    "8rRbFqb7u2smCxkxu",
    "f9vHd7txoW5tx4dhi",
    "KA3WoHn6YwFFWEKht",
    "EgJa5qZACZ9duzkXZ",
    "c3pxiSAy4bXupde3v",
    "P4ezian226huKkTys",
    "SZjbpkyNMAtm3MfxR",
    "tbdEmQi7fDXucfvxx",
    "vpBkZJ83zdH9sqg3i",
    "SQPF37N3NhKskwMpX",
    "nGap8FTe5D6iBKDyi",
    "M5tSKfyWoqEb54RQ9",
    "WqKaLrxN95E2QE7p3",
    "vG3si636dxf4WXgiW",
    "Xs6srzgFaM8WagwY9",
    "nv9Wofci2BGp5GQ9n",
    "G75RihDfmyDb8CWEY",
    "C8dzRjvr9ocwQXprZ",
    "HhisP2dE39gv2Rp5h",
    "geGCjcGwkdqdXh7wG",
    "Taqu6iLzTG5ep7fNg",
    "aNcejDE8xnbGefJSW",
    "qiTjbupD9WNiZMF5y",
    "Csfe644kfTH6dBSd7",
    "vRQCXdamEZ9WhXaoY",
    "2jgXMrLtMCTmob8Qc",
    "2fLy43ENMsZfCSbE9",
    "WNQ68Tsg4CX9hCu3r",
    "rnbGCrfTfuQq8uviG",
    "scZALQf2F2Cy3Cnkt",
    "JEWrk3PCecgnEBZZR",
    "DijNix9WhKA49oD5s",
    "7DhyZasDPmaRNfn9k",
    "wXghxfsvyjjtucSJL",
    "L6QoeJriobLg2o8Ke",
    "cF7hrmrdfwxsY4Lgn",
    "ri4AjyMgPzAcLTN29",
    "o24fEFE9ZJ4gY8RYW",
    "bN7MD3mDTP7okNw2S",
    "nCDbA3eniiKNmfhk6",
    "XwCkeKoSpqjpsjKRj",
    "jQsbDLymbcmPAZtAJ",
    "XEroW6xSWRxFA7wPS",
    "7NTzaEiuAMRR42G3b",
    "h47xFZmcnZiSLkMy4",
    "2Locnr97vCW4PdB5s",
    "iPzc9aovmngxMzC6b",
    "aEnbMTwEojdeaNQpz",
    "m2iHx2mD2KEuh37J6",
    "iKPpX9kNwSMqc6wHX",
    "kTDqPZjojtAD3gQBN",
    "QLGrwYErkhqGgmXkj",
    "PhpxqXWJg5QSD2Siy",
    "QSqqRLcFM9PZ2vFY3",
    "gXLkydYKxqwjhPpif",
    "NBPyuj7CFoqgifwLk",
    "2YacSsRBfmvxsJmhe",
    "hMvwKwQ26NDNeQTR2",
    "GmkJcruj5NuYTpxFH",
    "nBweL3b28CSYSeMCy",
    "fdDKPhwWMtuq78tqq",
    "KT6dr9z6io5C8Gsqy",
    "HnQpFBTdvR9CzmDQH",
    "uADYRapemJSGo8bfu",
    "g2dWhWD4qs6X8o7Ct",
    "ojA7qyJRbQeajqosv",
    "87sf6gnv5bneo555h",
    "tiCt5X3ueb4YQmGs2",
    "cXnar2hCyHu6D8SXR",
    "hrWsQwnPXFZ5RBny9",
    "RArfFAiHmcoNaveQR",
    "RxBZgKXToJLchDcgq",
    "EA49MxurZwsQnswj8",
    "TjLnerpFowNDTSNdb",
    "2EZygmPRRv8SwewD4",
    "rCJ2v5Bk8hHCC5Wmd",
    "FtGyiT97z9uCzHec4",
    "ReXrarrLLDRWGHeG6",
    "abQNiW2fAWRwDhc7M",
    "Q2QJvetGirHqWKQSm",
    "aeXEcBLBfnRm3jjCa",
    "HvGRcHNysKuyvvL9w",
    "HEhhz6sjgvoj5Nfuc",
    "2nvnK4uTGnFwT8we2",
    "jRbuXJcvmnotRiJxh",
    "KHedr2LAFkqcDHrAb",
    "cPcaMY9ggzvXd5LGQ",
    "CeCxuivsw7DBnE85S",
    "kcrtPr8h5KupkGcMk",
    "azZcBCSeF29FGHFZF",
    "89cmNwhjopLY5FMha",
    "3qDtnrrp8JvWtzBzC",
    "Lzev7hu4qXPBTLwgr",
    "EqBpmpnsoi3h7Dt3G",
    "AfW4DbcGnKEA3aLaZ",
    "k6ejepkByYnEh3ixZ",
    "ttcX3m7bvikWcujca",
    "2YapFi9FttC6opHM5",
    "7NSRPKG8r62gbGrJD",
    "8kkjyREnqR6HFfMAZ",
    "n7fLapqfQggaqYyRE",
    "qcN22j3w4f4Fk792P",
    "XkaDFEJcKs7xMDB4k",
    "eqT5vMHcYkn69mXmb",
    "2AZuRrmHaFt9Rm7Hs",
    "KDSvSFmGtgMj3FtsQ",
    "dpK8K22hWFmYEFu6i",
    "jP3vHAththRrSkove",
    "PpAfhbqfP4oTmqY8M",
    "kmq6WTcx2KvN7cDRD",
    "m2pej9CwK2SiRd6pP",
    "pTRmWxXBTPyMvS9eA",
    "dfEFbbEvYt3ECgXqe",
    "HJKiSdx3M454SLkDq",
    "5n6BuiL39sn7ccc2M",
    "XX8mRMdwzQXHkQobo",
    "4Sz7T22BWN2LxHTSv",
    "c2jqxhsgnht6kunu5",
    "Zi3EAATti52D3Xxd8",
    "HrGf9vbK2eZwAbpHw",
    "2KbCPm7NsssH8NaKX",
    "DMZbhn4i2nqiLXJQP",
    "yYAsuZ763kDmoYQDB",
    "YQbzHnrqq4Ja5tSNX",
    "QY2TzEgpEr4e8NLcK",
    "a3a9tyt92W3KdyNMH",
    "4rGwMiG34JTZxhQfu",
    "Anyn2fpB5xmScq6NZ",
    "jpCEbCPnNuB96mJ2g",
    "FfrXmej83cooC5HEw",
    "rYPsejdBwg8n5aS67",
    "nnd3TvJQBxWWhAYRa",
    "pCHTXJbqWysyurcAg",
    "NzuKLbJTiWB9ndSE2",
    "qJCFpEDZeJhpCnZwh",
    "rY4sx4eLbBasEKPN9",
    "EwJKS6EzrgeadiEi7",
    "Ahg9N8gkDp5RoPNu4",
    "swMGaj5MKGwqv2TZg",
    "uBZecz7RrNi8Zeodh",
    "LYtiHsabZWja7WzwR",
    "x6Bjj7wtRymbhokyd",
    "Jx267w4mLFtgeFto7",
    "PT7XtyTYTtJBmb8Eo",
    "dfW385y5NcAvoJhDz",
    "MiLrLB8gZwE8gvo9s",
    "uCHjbsKzZ8ks5WCLM",
    "9c522ELmNAFzu3YRp",
    "pfMeHxvg8KzFubJB4",
    "wKiMKFxqAEE3TKQtn",
    "SAvqeaM2r3oR45LxF",
    "Kutasn7Ru7m3XJKhy",
    "fRTmRxQkiG9wRc5Mj",
    "JCMiFNJrKma5SzkXx",
    "6pNypxd4usa87R8wH",
    "cErqX7RYbS9XfRogT",
    "fQEmgcJRSqEMW6KLB",
    "DAaFsy3chZfgXfY3X",
    "mmTYAjgdArhuJhzLZ",
    "St8LaeJYtj9AdcjYD",
    "6wHDmEc6KqcX5Znhw",
    "MzSQjxSnNtnuJ7Nqx",
    "vQm6BBHoypDmzLQ9z",
    "AXCweh5EBkzj8HX63",
    "iS5b3q7iFkTXrnAWA",
    "Rz995vscyK6qjQTkX",
    "QZsst4kNJiHTySkqS",
    "CXLaEAGC2a8quAcuh",
    "BmAYdp2YfDH7oFXEX",
    "r9NGyLjpCQmJX3weX",
    "pz3f56twRGJRNSPLM",
    "DDopj2za3W5c5fscr",
    "fAZKPTXy7JrhJM9qt",
    "qTBNnxYzoLEp7NQQ6",
    "T8czgZKpssneeRwtf",
    "rajbYYK5Cnih7Hj2s",
    "c9zojonwgeBy3zK5Z",
    "7ZrvPC74ZzZRFXtJ5",
    "PhYPnwZBdsfQ4phB7",
    "Fd5XCYQuYG5mRXyZd",
    "HKDQ6MZd7HWddB54h",
    "w2QTgkEyS5upTmMiu",
    "epWc2qPoeKDekroeE",
    "dBurfR5GW3hYJZJvE",
    "hfhpYLvRHqJNTfros",
    "v98BqG24AbaKXRLpS",
    "vpBzoDrL598KpY4vH",
    "baPahTqouJ2bXrJp9",
    "vvjSt9wMFTGPAhDZy",
    "tzisW7JJzGH6Es9aC",
    "zYKK2i7MEDmZrm9f8",
    "JvptPkY5MbBQzFMTa",
    "WTCMJRroChW4paFuG",
    "hz4LivqoE2f5AfPQ4",
    "rgptnYhX8vfKFh6k2",
    "eTYeYcSyScj4y759D",
    "fkPsNGMjCm4kL3XBF",
    "RPH95XAgfZ36k8hit",
    "CPHcaPawAsgMmp4NX",
    "vd3Mn4RZkALGHBECM",
    "y5eMWSnxW5JxD2kme",
    "SnkAXYmGg3irdqP24",
    "eHXdrjtgMg4iinqiE",
    "Quhay9vp5TnNjroiW",
    "GkbCq3MwTGafTyHiE",
    "FWrRahHwoAmdXqToY",
    "Dp5YBa8uqr4CPDoQE",
    "Xv5XmgA4YhzsTHTmQ",
    "yK8hYqDS4ZTKPsXzf",
    "aLF5Wr9AZ2kimfFZA",
    "DvZzchwDQjG8uL2FL",
    "je8TD9d4usTmZXGJ7",
    "2iCm9rYoBW6JFLNLf",
    "TgtPgg7tLobi77ZTY",
    "NtkioFa2DfDFPY3ap",
    "6SuBRnwpr7qACZ7DR",
    "BCHDu5Roy57StQe6X",
    "Jc2ZLR4STnXN4biqx",
    "Af42qSjwBxtYuJZej",
    "DbZjL3pDBgoPsj49B",
    "QX9B9GMaX6wgqnKGJ",
    "HtkAXvbpHZwMYcXBp",
    "mrTy8T8WbWRfD9DKP",
    "izPt2v7wRcsDfz4Fx",
    "hGnrg8Cr5HXRFaWYy",
    "tY3yosez8rgf87PuG",
    "FER2FyttypMrT7vcL",
    "c884mQnHtDzLtCF6f",
    "2Eib58qov6Mi9h6s3",
    "MaMZas2eGnqShPQAh",
    "Djx4wuhGznCEZwKmj",
    "ERSE4Za8SnZ5vtDSQ",
    "sXMJ4fiMJXygHQeJX",
    "MwRzGuip5JqidBkEs",
    "w6ujJPaMZNQA67NQ5",
    "C2MkaY6sre7PTrgSs",
    "FmqKfWR9cKYcyZF4j",
    "6CDsgTrxjTyX9s9Qy",
    "JPyuTDFjiDdFSLMrP",
    "5RkXELQNSQZ77Zb5K",
    "YrXRJ8doNBugi2LtM",
    "RDkSkoaPQiqq9NbfS",
    "mv9yMjYrwQBXibYZr",
    "kirjup5j9ebr2NmSp",
    "xxXAuMhjSK6TJ8duY",
    "f9SetYciTnrAgDzHQ",
    "WC5xoibzxkJuXB5ds",
    "gFqtHaTF2ooEGGoCX",
    "KmD8N2MiBBqk9z8eR",
    "RfsjmCXcjD7A8JJg8",
    "JSXcvkbyTuYbpFqvv",
    "nJhzuXRQzTjXohCaW",
    "oYq7A4NqRnK9bfBYv",
    "oWoF82ipJv5AWohee",
    "ypLo26qPLyX8GD8m3",
    "hA3zzwiEKgFj2rDi2",
    "XrtvofW8ReZp79GCi",
    "FXDsjL3nAL3C9Yvgt",
    "6yTgnc6z5S8XBx7Pw",
    "CJWscg2QoAG6jfNEL",
    "G2QkM6b7Z8qLiRLn2",
    "mYaew3KFYshznWGnE",
    "NJCwqhFNahN8vrCrg",
    "Sg5QTtNkTgxqMo5Kh",
    "7JzmkiPFy5d3PJhKh",
    "SjYxYxNcw3yPF5t7R",
    "zg6m3umTdiywpKnxd",
    "cChkqYw7bd8HhmB5D",
    "B3z9rNugcqJgsFhQm",
    "o6dkJ8C3bW32cZjdS",
    "oQaMo8sdsb6dPdND7",
    "mvkFHt5k4LMik4s2p",
    "8XSNG2neqSybBW4em",
    "xoJ3vNAFm3BNbtjGr",
    "EX2G72HZZugPeKqmB",
    "29TPDcqGXjePCztEN",
    "bxhkrPGcqdsjAn5Rp",
    "j2F2ik7s9NBvLa5QN",
    "cCgpfTKY86tuic3uY",
    "fvvJpYioXH6xGpSZ3",
    "aRkdrL3to9toxDyim",
    "spizTqvYwni3toh2b",
    "SBP9i9Fjaz8f3Z9Tp",
    "5E2JGm4nbFLxWWigM",
    "pQ2v9PBBvSea2Ajv3",
    "5ZySvwQNQEGvYECob",
    "QxGwwsRJSRC68sG5r",
    "Tqm5GaZGR5eJTzkjT",
    "KgEL6GLCmwbgxRDuL",
    "oau8fAZeCSaapywhj",
    "YenZvPHmWBgfcQnfS",
    "9LNCcrseKXuwCZ5cv",
    "QfHdQ3Jaw5HyM4MmR",
    "S2b4Hjo7Zkc44bYGP",
    "xoduempdCdSPkuEyu",
    "mfEFRJjPgERGXWaee",
    "aZTnm65BdtEDJ5pra",
    "aZwjMkaChGj7tpsDa",
    "22MdGe6abtxM7pG2w",
    "XdLHvpMSAhcwZr297",
    "JXY3GBDwjpg86eHN8",
    "EhskYN3A52MoxbgZp",
    "8fwPGq93G8aCiAnM7",
    "tvun4o5JMQ2aGkyeq",
    "NaiRiLayBpymhz23N",
    "BWGRafnJGDC2KsuKf",
    "uYsQJdhojkbmTrEC9",
    "7TJJCmNNdXJXM95E7",
    "fEjg23Q55xqAQ638F",
    "9QjDXWGCensWZwW34",
    "x2afRd9BDFthBCm9z",
    "6w6XmMpKKTZW7F5bS",
    "kc63wh789RrW4HBNG",
    "dsGCGbJJwkuSDHhuk",
    "gre8FQAZd9P7jA8ot",
    "xbLyH9BTjencGwy49",
    "83NHY3NaB3PoMLqXm",
    "xFMqsb3iEm4D9RwuD",
    "D5FWrLCRcEqNDY3WT",
    "iWgjzR85xNmd8m7yx",
    "ejRQsYPx9YWdnwLMC",
    "Ho4SJ6QJ2csBpFE7e",
    "YgbSrNbb87rs8GCRr",
    "jEXqQGEgvBEd52tcL",
    "vLK6XNgMAFvMhvfhh",
    "HsX3La6LSALgeCFpA",
    "hBwqmtQEooMws9fu9",
    "Q9rG4vT7ENgDudqTR",
    "c2Sef3TFeZPF2bZhT",
    "XWSa3nmjMkHpgGZnq",
    "rkpd7gKbhMBbkt9vh",
    "htEnE59NALhmEE3Er",
    "KbnvD9aJMWsvf4PWv",
    "J68eZGEiWZEEp6cdF",
    "cQnfRd2rtNZyWvDqZ",
    "PWqPsBwMuPdQ7Cs42",
    "kyAYuuSZ6PMxwdJ95",
    "XGCyrFSQw2zm2Yq67",
    "MGNX2YnAno7dumGJh",
    "9pbfpB8b4FB93EHRK",
    "cSP83Hb9MXLTtPdwB",
    "8C7Efia6unpEB9YiB",
    "94ShPadaSM5RLc9CS",
    "N3SdvbDNrcMrGJaYr",
    "3zWsQXjSXLLhdkD55",
    "nFYjzM5sFYMygwapM",
    "ezHvKuWLLoKmTtikB",
    "S8xcwDfML2B3kDnre",
    "F8BpB2gp7YxbFyhLM",
    "xugCYFDKMWLZqkBDT",
    "qM8zufMaSGDeZJ4xx",
    "yeM82jMhoW5YZwJu5",
    "bMjAbYHLS2cEf2HBg",
    "vZQ2g6qhqoAJ4D6P2",
    "wixiPpFzpJzsWujPP",
    "QH9pqafP9QAiMA5hX",
    "Ky7rR3YEfXZvYoLP6",
    "t2PqxxDCYKhsLGquM",
    "QECaNakDNGepNBa9e",
    "3C4jNNiCLqmrzbTBc",
    "4NggJ5MxTXoueMzXK",
    "zQ6viAryNJqcJh4cY",
    "eG8WmjPoBZrdXobKa",
    "Hv9NMjNQWecgdAvmR",
    "WPvgjr2YRHxEnucac",
    "5AH3MRTBWEkXjYejs",
    "RKTuaefx7jZYYmNDG",
    "EyhBuGzAQm72fqufN",
    "B7BuptYhyQ6ocACwA",
    "pyxExLos9y8akqoQs",
    "e2bveA58BJdeakbR5",
    "KM4aJGuScRbAdnM8S",
    "mNcsmNtgHpZY6rwCW",
    "rGkmRdLkjJZZ9Gtx8",
    "tEWRbywr5qsHBo36p",
    "8uAb4SMNjWKYgbBin",
    "BSv7ZTcNcAp4kKTQQ",
    "JBaF37bssvNN8gpc4",
    "a6fnCWWhZez8rL4rP",
    "kts8tjr7ixHRjRYzK",
    "Mo9LoJxEorbD9Tkvs",
    "JE7qcvCb8ZpRtFYEp",
    "apCNgKfqAF9xuw9Gu",
    "gngm3nLsWkHiATx6P",
    "4L3ttckQKqfjm92cj",
    "SXbruLTjxDRMaxtNY",
    "NYsfwGwF5EpFo4pQT",
    "mvEYznQGK8cW8DYCf",
    "Yz8nBzQXuSpdpZrnQ",
    "j9NgbS5YqPW7g9Myp",
    "BXkG2Sk452F8XqeiN",
    "3g388CJbHM9GZExB4",
    "ouCkYKHvh6hKD4vhg",
    "rDufwvTML75o6AN5M",
    "tX5NhwRB5iEojG6Jk",
    "46F6Gp6dBETADBvNf",
    "qD66vaWrqeGxNwxNF",
    "hdqL25rE5tjxAuwqp",
    "Gprbca4Fa6LH75Xrq",
    "PisRL79gTuxZpr4gy",
    "ytMt5DvX8QScsDSGx",
    "3n4abBgPnWd5gk5G2",
    "E6mjfdSBAdnnxWj2D",
    "ZnLz72HKwzzAcKDN8",
    "uj8wWtMK5jyG5gDpG",
    "fpoWK45TKGXtKxj5j",
    "jF2gezx6xozwXQenZ",
    "CapcAfaSBmcRxgTae",
    "5k4hSE2aLhd5K2npi",
    "autmHB3FRztp79ZyA",
    "d35sgaCtzzN9RRaBh",
    "dmk9Wg6JmWd4Mni5W",
    "CFGR4HkkRD8PohSJg",
    "9yoQCvHwFG6gRdCjQ",
    "qwpwb5yWz5cpEBps5",
    "G7JNYmqY8xAfQznRA",
    "pgLfDdydBKSMuxGNj",
    "Wne6redeGregPsmSp",
    "bMiXWdvhhkKPJX98c",
    "nejPef3rjpk7Qn9KX",
    "6uiWCjgtCbfpQo8fh",
    "cSxqXA4RLCMF2fXuP",
    "uTQ9QYWFTjba6GiJR",
    "NppqPGv3MseTeQPcx",
    "B54PfZnfiPvqqDzgr",
    "DWZFYHJTGoWDacTTE",
    "xFnR34HFHdZ8pZFFN",
    "sQh2o5LckSKSjXvCe",
    "zLczGjHhEiRWCZqLH",
    "eH6ZPB45fmETEz3Lq",
    "juqsFty7m488ukcbT",
    "xcXSAGP7H2rTmtTF3",
    "JtcLY9JgeP33gPN74",
    "g9LdQZnCbMfvAnLXc",
    "2u4FsnjxDrLvPQfgX",
    "9G4biom35gXkfB9Zd",
    "nRT6DGZm3tK9Pnq4h",
    "Tcn6ntGf5dqFbKkMt",
    "RBXwLZTxvQDEu7e48",
    "mNJPv2Av6R62RzzDW",
    "EGzuNQQjjTjRtGhL5",
    "QMJBo4EXc9LCbWP7d",
    "ajWoCWm3wCFZmWeKe",
    "m49TiuwGf2xZjFzw9",
    "2YjLw9E8q5ahSRnRk",
    "bZoKyf7QCjxvTuByJ",
    "pQNXuMMEZcNNnwZBx",
    "oJrc5paNkQZnkFg4W",
    "oH7zubtr7hHdn8qqh",
    "vxNAwusHckxkGGads",
    "Zb88FunDpoRDZD3fw",
    "PxRWdQiD3WLRKB6QB",
    "WsenvNXXxHrFHmb8s",
    "zcdjNcDzKKY6qkMEf",
    "mL2B7gzuTa9vv9qoA",
    "fbaqpSLGnBwnY3jbe",
    "DDSm6MyoDAF7sAJva",
    "4vNzHkeisMCH2vMPY",
    "TSNtGhq2ec4T8iS7s",
    "nAiwYeXXXbBAYrrtY",
    "TQjhmFdAwNeWdjX5k",
    "34zaRnGBga4zgCZa2",
    "PntEL5XDNQe7bb3Lt",
    "4xi4RtMW82bJY7m3S",
    "SybSbzEYruHa9fiyb",
    "CrHLuP3EgjScga9mv",
    "8wt6u3TpZuw35MFKD",
    "cR4XacYDiW87ciGbt",
    "xEdexaqiHuJXCHXBd",
    "XNcc9et9cF9ygKgv7",
    "EqbLNfkWmtB2sjKku",
    "LijxitSzR7kdZyZ82",
    "6qkPgQxgLDFB2fEtp",
    "hYzLjyuvygWx6y839",
    "9u5b9HCy6jKPNkScJ",
    "AZWF9gFKSduFxnPw3",
    "FhCXksr2fCqYWWr54",
    "G6eYnrekK4f5NvsLK",
    "sJCMhGwP5PSNQAc5a",
    "bS9j7aqh7wERofQ3Z",
    "y3dtEGn5PwN46mnGt",
    "XqtnEY4tN8RubQ9vL",
    "wX3Bjj2QiY6kzX9Xd",
    "Aasfck5PxkKfGAYLA",
    "5QrP6bKGi6kQDcczp",
    "QSuBn3c7WY43aP3MR",
    "actHP29jabNBKeZ57",
    "8nSWqx6ArEen7eQoq",
    "3pSDtCKTmmNLauoWv",
    "WgSYz48gvneg4T6M7",
    "Kt3xMyww6Z7dzFBuy",
    "9jeLHBKciGYH9E82k",
    "9fuXsNGxvD5BYC97c",
    "FHk485ReDA7wLMJRB",
    "GF9Dxr3JDPyqrskEm",
    "w4meS9Ms5dpTZxB6i",
    "YHjZ7BsTscanZWfi5",
    "RZmJtLLKg6Jn7JXTo",
    "WtwmMWT7mDXH2xAPi",
    "uJiBuR3hDEssb8M7M",
    "gBpEeHP7PWPAMc7Q6",
    "M93Jsg3Ms9ue53HfD",
    "jXwjRbnJsdXuzngzr",
    "5tzwyeGQ7DpzBeWgq",
    "uFc3Sc375quJimiyk",
    "qP6HEYJc2pGPEhPzh",
    "bHcsqJrMPqaZb7XWP",
    "4yLFzRXnWBfSQ7ri8",
    "FGmfwK27LLQzsR4dK",
    "jsEMBBPfggD4GTXXH",
    "d6kXeHzs6Z8Wwthnr",
    "k8v75C9TngAJAe4Sr",
    "semPJrieohdLpLhQD",
    "wARJYe3JMbHjghagj",
    "sCMfnArc25xGYiMA4",
    "du6jm4kBrPCh7iz6d",
    "8jj8AEjJ8rBqt3tFs",
    "aWQFJk3ZypxN9ccaL",
    "BQu2vmdh7MJgubaxC",
    "o6YiG9Crnkiropqj7",
    "uEdpsTA7FmHZEXzw2",
    "mx8Pg4FCi3WqEv9Nu",
    "q4B77s8evWx8qchB2",
    "YnBmonht7yQC6BYSB",
    "8pgdSgZZtmbP6y4Au",
    "reog4PpMm6bNF4WBz"
]

# download each id
for imageId in ids:
    url = "https://hot.nyc3.digitaloceanspaces.com/normal/" + imageId + ".jpg"
    print(url)
    r = requests.get(url)
    open(os.path.join('training/train/maps', imageId + '.jpg'), 'wb').write(r.content)