"""Draft keyword lists, one per category. They are a STARTING POINT to be measured, not final rules.

words    : matched ignoring case, whole words only (a plural "s" or "es" is allowed)
acronyms : matched exactly as written (so "AI" does not match the word "said")
Step 4 measures how often each word appears and which ones are noisy; later steps refine this.
"""
TOPICS = {
    "market_stocks": {
        "words": ["stock market", "stock", "share price", "shareholder", "wall street", "nasdaq",
                  "dow jones", "s&p 500", "earnings", "investor", "bond yield", "treasury",
                  "inflation", "interest rate", "federal reserve", "central bank", "dividend",
                  "hedge fund", "bank", "banking", "recession", "market cap", "trading", "profit",
                  "revenue", "economy", "economic"],
        "acronyms": ["IPO", "GDP", "ETF"],
    },
    "technology": {
        "words": ["artificial intelligence", "machine learning", "chatbot", "openai", "chatgpt",
                  "software", "smartphone", "iphone", "android", "cybersecurity", "hacker",
                  "data breach", "ransomware", "robot", "semiconductor", "chip", "nvidia",
                  "microsoft", "google", "apple", "startup", "cloud computing", "laptop",
                  "algorithm", "app", "tech", "technology", "gadget", "operating system"],
        "acronyms": ["AI", "GPU", "CPU", "VPN"],
    },
    "health": {
        "words": ["health", "healthcare", "medical", "medicine", "disease", "cancer", "vaccine",
                  "hospital", "patient", "clinical trial", "drug", "covid", "virus",
                  "mental health", "doctor", "surgery", "diabetes", "obesity", "outbreak",
                  "symptom", "therapy", "public health"],
        "acronyms": ["FDA", "WHO", "CDC", "NHS"],
    },
    "energy": {
        "words": ["oil", "crude", "natural gas", "electricity", "power grid", "renewable",
                  "solar", "wind farm", "wind power", "nuclear", "coal", "energy", "petrol",
                  "pipeline", "hydrogen", "power plant", "fossil fuel", "gas price"],
        "acronyms": ["OPEC", "LNG"],
    },
    "politics": {
        "words": ["election", "president", "parliament", "senate", "congress", "government",
                  "minister", "prime minister", "legislation", "lawmaker", "vote", "voter",
                  "democrat", "republican", "trump", "biden", "diplomat", "sanction", "policy",
                  "white house", "political", "politician", "campaign", "supreme court",
                  "foreign policy", "treaty"],
        "acronyms": ["GOP", "UN", "EU"],
    },
}

# Decision (user): crypto belongs to `technology`, with the sub-label "blockchain_technology".
# It is measured separately here so the sub-label can be filled in later; it is not a 6th category.
WATCH = {
    "blockchain_technology": {
        "words": ["bitcoin", "crypto", "cryptocurrency", "ethereum", "blockchain"],
        "acronyms": [],
    }
}
