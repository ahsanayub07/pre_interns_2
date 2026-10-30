"""Rules v1 for the local (no API) classifier. Everything tunable is here.

Each topic has terms with a weight: STRONG (specific, hard to confuse) or WEAK (generic words that
appear in many unrelated stories). Acronyms are case-sensitive. The domain is never used.
"""
RULES_VERSION = "rules-1.3"

STRONG, WEAK = 2.0, 0.75
TITLE_MULT = 2.5          # a keyword in the title counts this many times more
LEAD_CHARS = 1500         # characters of the body that are read

# ---- decision thresholds ----------------------------------------------------------------------
MIN_WORDS = 25            # title + lead shorter than this -> uncertain (too little text)
FOREIGN_MIN_RATIO = 0.10  # share of common French/Spanish/German/... words needed, and more than English
FOREIGN_MIN_WORDS = 20    # at least this many words before the language is judged
MIN_LATIN_RATIO = 0.70    # share of Latin letters; lower -> another script
OTHER_BELOW = 2.1         # best topic score below this (one lone mention) -> other
MIN_SCORE = 3.0           # best topic score below this (but above OTHER_BELOW) -> uncertain
MIN_MARGIN = 1.5          # best must beat the second by this much ...
MIN_RATIO = 1.3           # ... and by this factor, otherwise uncertain (ambiguous)
SECONDARY_RATIO = 0.4     # a topic with >= 40% of the best score is listed as secondary
BLOCKCHAIN_MIN = 3.0      # crypto score needed to add the sub-label blockchain_technology
REVIEW_BELOW = 0.6        # confidence below this -> needs_review

def _t(strong="", weak=""):
    return ([w.strip() for w in strong.split(",") if w.strip()],
            [w.strip() for w in weak.split(",") if w.strip()])

# Template pages that are not news. Matched on the ORIGINAL title (case-insensitive).
COMPANY_PROFILE_RE = r"company overview|company profile|stock quote|stock price\s*[,&:]|stock price (?:and|&) "
DEALS_PAGE_RE = r"discount codes?|promo codes?|coupon codes?|coupons?|voucher codes?"

TOPICS = {
    "market_stocks": {
        "terms": _t(
            strong="stock market, stocks, stock price, stock exchange, share price, shareholder, "
                   "wall street, nasdaq, dow jones, s&p 500, earnings, investor, bond yield, treasury yield, "
                   "interest rate, rate cut, rate hike, federal reserve, central bank, dividend, hedge fund, "
                   "market cap, recession, inflation, bull market, bear market, quarterly results, "
                   "commodity prices, forex, mutual fund, private equity, venture capital, merger, acquisition, takeover, "
                   "tariff, mortgage, credit card, loan, stock, trade war, trade deal, price target, funding round, valuation, "
                   "bankruptcy, insurance, insurer, bitcoin price, price of bitcoin, crypto price, crypto market, crypto trading, "
                   "crypto exchange, token price, trading volume, bitcoin etf, buyout, portfolio, investing, investment, "
                   "pension, retirement account, jobs report, unemployment, cost of living, consumer prices",
            weak="bank, banking, economy, economic, revenue, profit, finance, financial, trading, market, markets, "
                 "billion, layoffs, retail, credit, business, businesses, ceo, chief executive, sales, deal, analyst, funding, "
                 "pocketbook, consumers, shoppers, employer, hires, appoints, price, prices"),
        "acronyms": ["IPO", "GDP", "ETF"]},
    "technology": {
        "terms": _t(
            strong="artificial intelligence, machine learning, chatbot, openai, chatgpt, large language model, "
                   "cybersecurity, hacker, data breach, ransomware, malware, semiconductor, chipmaker, nvidia, "
                   "robotics, robot, quantum computing, software, smartphone, iphone, android, operating system, "
                   "cloud computing, data center, open source, algorithm, bitcoin, crypto, cryptocurrency, "
                   "ethereum, blockchain, stablecoin, nft, earbud, headphone, airpods, smartwatch, wearable, processor, "
                   "motherboard, firmware, overclocking, graphics card, amd, tsmc, smic, qualcomm, foundry, "
                   "lithography, nanometer, 7nm, 5nm, 3nm, gaming pc, gaming mouse, gaming keyboard, gaming laptop, gaming monitor, virtual reality, augmented reality, "
                   "vision pro, wi-fi, oled, starlink, fitness tracker, smart home, smart speaker, wireless charging, dash cam, dashcam, "
                   "photoshop, adobe, browser, tablet, ipad, macbook, fitbit, garmin, kindle, foldable, charger, battery life, "
                   "spacex, streaming service",
            weak="google, apple, microsoft, meta, app, tech, technology, gadget, laptop, startup, internet, "
                 "digital, online, social media, chip, samsung, amazon, xbox, playstation, nintendo, console, "
                 "camera, drone, cooler, gaming, windows, linux, chrome, tiktok, instagram, youtube, benchmark, intel, twitter, facebook, whatsapp, snapchat, reddit, keyboard, headset"),
        "acronyms": ["AI", "GPU", "CPU", "VPN", "iOS", "LLM", "SSD", "USB", "HDMI", "5G"]},
    "health": {
        "terms": _t(
            strong="heart attack, clinical trial, vaccine, cancer, disease, medical, medicine, pharmaceutical, "
                   "patient, surgery, diabetes, obesity, outbreak, epidemic, pandemic, covid, public health, "
                   "mental health, symptom, therapy, diagnosis, dementia, alzheimer, healthcare, health care, "
                   "drug approval, virus, pregnancy, medication, prescription, pharmacy, eczema, psoriasis, allergy, allergies, "
                   "asthma, arthritis, migraine, infection, flu, antibiotic, cholesterol, blood pressure, anxiety, "
                   "depression, immune system, gut health, sleep apnea",
            weak="health, hospital, doctor, drug, treatment, clinic, nurse, wellness, vape, vaping, smoking, tobacco, "
                 "nutrition, diet, abortion, sleep, calories, supplements, vitamin"),
        "acronyms": ["FDA", "WHO", "CDC", "NHS"]},
    "energy": {
        "terms": _t(
            strong="natural gas, crude oil, oil prices, oil, opec, electricity, power grid, renewable energy, "
                   "solar panel, solar power, solar energy, solar farm, solar plant, wind farm, wind power, nuclear power, nuclear energy, nuclear plant, coal, petrol, "
                   "diesel, pipeline, green hydrogen, hydrogen fuel, power plant, fossil fuel, energy prices, refinery, oil and gas, gas prices, "
                   "fuel prices, jet fuel, petroleum, oil tanker, power outage, load shedding, power cuts, blackout",
            weak="energy, nuclear, power, battery, solar, hydrogen"),
        "acronyms": ["OPEC", "LNG"]},
    "politics": {
        "terms": _t(
            strong="election, parliament, senate, senator, congress, congressman, legislation, lawmaker, "
                   "prime minister, democrat, republican, white house, supreme court, referendum, sanction, "
                   "diplomat, diplomatic, treaty, ceasefire, peace talks, voter, vote, coalition, cabinet, nato, kremlin, hamas, hezbollah, foreign minister, foreign ministry, "
                   "defense ministry, defence ministry, armed forces, missile, airstrike, peace deal, national security, "
                   "bilateral, summit, delegation, envoy, state visit, official visit, naval chief, army chief, chief of staff, "
                   "chief minister, provincial assembly, national assembly, militant, rebel, insurgent, terrorist, terrorism, "
                   "taliban, houthi, strikes against, military strike, regime, annexation, genocide",
            weak="president, government, govt, gov't, minister, ministry, political, politician, campaign, policy, "
                 "party, mayor, governor, trump, biden, war, military, army, troops, russia, russian, ukraine, "
                 "ukrainian, israel, israeli, gaza, iran, iranian, border, migrant, migration, refugee, opposition, "
                 "ambassador, embassy, consulate, immigration, deportation, asylum, pentagon, navy, defense, defence, weapon, "
                 "syria, syrian, yemen, lebanon, lebanese, iraq, iraqi, afghanistan, kashmir, taiwan, sudan, libya, "
                 "palestinian, west bank, provincial, assembly"),
        "acronyms": ["GOP", "UN", "EU"]},
}

# Crypto words are part of `technology`; when they carry the score the sub-label is added.
BLOCKCHAIN_TERMS = {"bitcoin", "crypto", "cryptocurrency", "ethereum", "blockchain", "stablecoin", "nft"}

# Evidence for `other` only (never a category of its own): sports, entertainment, lifestyle, crime.
OTHER_HINTS = {
    # sports
    "football", "soccer", "cricket", "tennis", "nba", "nfl", "mlb", "nhl", "premier league", "tournament",
    "championship", "olympic", "coach", "goal", "penalty", "striker", "match", "league", "season", "rugby",
    "golf", "baseball", "basketball", "hockey", "quarterback", "goalkeeper", "racing", "fans",
    # entertainment and culture
    "film", "movie", "netflix", "album", "singer", "actor", "actress", "celebrity", "box office", "concert",
    "trailer", "episode", "horoscope", "obituary", "puzzle", "wordle", "crossword", "festival",
    # lifestyle
    "recipe", "travel", "fashion", "restaurant", "wine", "wedding", "dating", "weather", "holiday",
    # crime and accidents
    "police", "arrested", "murder", "suspect", "stabbed", "shooting", "strangled", "jail", "crash",
    "accident", "killed", "died",
    # science and space
    "nasa", "rover", "species", "archaeologist", "solar system", "astronaut", "comet",
    # video games (news about games, not about gaming hardware)
    "video game", "gameplay", "gamer", "esports", "game",
    # natural disasters
    "typhoon", "earthquake", "hurricane", "flood", "storm", "wildfire",
    # motorsport and science (outside the five categories)
    "formula 1", "grand prix", "telescope", "planet", "galaxy", "asteroid", "astronomer", "dinosaur", "fossil",
}
# Violence wording is weaker evidence for `other` than sports or crime words, because war and government
# news uses it too (v1.3: each distinct word counts VIOLENCE_WEIGHT, not STRONG).
VIOLENCE_HINTS = {"attack", "injured", "wounded", "casualties", "bomb", "blast", "explosion"}
VIOLENCE_WEIGHT = 1.0

# v1.3 finance focus: when the TITLE shows that the story is about money (stock moves, ETFs, earnings,
# deals with an amount, crypto prices), market_stocks gets this bonus even if the company is a tech company.
FOCUS_BONUS = 3.0
FOCUS_TITLE_RE = (r"\b(?:stocks?|shares|etfs?|earnings|ipo|investors?|investing|dividends?|price target|market cap|"
                  r"valuation|funding round|bitcoin price|crypto prices?|rate cuts?|rate hikes?)\b")
FOCUS_MONEY_RE = r"[$£€]\s?\d[\d.,]*\s?(?:b|bn|m|k|billion|million|trillion)?\b"
FOCUS_DEAL_RE = r"\b(?:deal|acquir\w*|acquisition|raises?|funding|valuation|settlement|invest\w*|buyout|bid|takeover)\b"
FOCUS_ASSET_RE = r"\b(?:bitcoin|crypto\w*|ethereum|dogecoin|doge|xrp|solana|stablecoin|stocks?|shares)\b"
FOCUS_MOVE_RE = r"\b(?:soar\w*|surg\w*|jump\w*|plunge\w*|rall\w*|tumbl\w*|slump\w*|climb\w*|drop\w*|fall\w*|sink\w*|rebound\w*|record high|all-time high)\b"

# Removed from the text before topics are scored: these phrases contain a topic word but mean something else.
MASK_PHRASES = ["solar system", "solar eclipse", "nuclear family", "tear gas", "power couple", "power play",
                "bank holiday", "food bank", "blood bank", "river bank", "stock footage", "in stock", "out of stock"]
OTHER_DOMINATES_MIN = 4.0     # other evidence must reach this, and 1.5x the best topic score

# Common words of other languages that are not English words. Used to spot French, Spanish, German,
# Portuguese, Italian and Indonesian text by what it contains, not by what English words it lacks.
FOREIGN_WORDS = {
    "le", "la", "les", "des", "une", "est", "pour", "dans", "qui", "sur", "avec", "pas", "aux", "nous", "vous",
    "cette", "sont", "mais", "ont", "et", "en", "de", "du", "un", "au", "que",
    "el", "los", "las", "una", "por", "para", "con", "del", "como", "pero", "es", "su", "sus",
    "der", "das", "und", "ist", "nicht", "mit", "von", "ein", "eine", "auf", "den", "dem", "zu", "auch",
    "uma", "dos", "mais", "foi", "são", "não", "il", "che", "della", "nel", "sono", "anche", "gli",
    "yang", "dan", "untuk", "dengan", "tidak", "ini", "itu", "dari", "pada", "akan", "adalah",
}

STOPWORDS = {"the", "and", "of", "to", "in", "is", "for", "on", "that", "with", "as", "was", "at", "by",
             "it", "from", "this", "are", "be", "has", "have", "said", "will", "an", "not"}
