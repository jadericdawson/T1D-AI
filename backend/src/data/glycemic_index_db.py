"""
Comprehensive Glycemic Index Database for T1D-AI

Based on:
- International Tables of Glycemic Index and Glycemic Load Values: 2008
  (Am J Clin Nutr 2008;76:5-56)
- University of Sydney GI Database (glycemicindex.com)
- Research from Harvard T.H. Chan School of Public Health

This is a LOOKUP TABLE - faster than AI calls for known foods.
Unknown foods fall back to AI prediction.

Format: food_name -> {gi, is_liquid, category}
- gi: Glycemic index (0-100+)
- is_liquid: True for drinks (40% faster absorption)
- category: Food category for logging
"""

# GI Categories for reference:
# Low GI: 55 or less
# Medium GI: 56-69
# High GI: 70 or more

GLYCEMIC_INDEX_DATABASE = {
    # =====================================================
    # BEVERAGES (is_liquid=True - 40% faster absorption)
    # =====================================================

    # Fruit Juices
    "apple juice": {"gi": 44, "is_liquid": True, "category": "juice"},
    "orange juice": {"gi": 50, "is_liquid": True, "category": "juice"},
    "grape juice": {"gi": 56, "is_liquid": True, "category": "juice"},
    "grapefruit juice": {"gi": 48, "is_liquid": True, "category": "juice"},
    "cranberry juice": {"gi": 68, "is_liquid": True, "category": "juice"},
    "pineapple juice": {"gi": 46, "is_liquid": True, "category": "juice"},
    "tomato juice": {"gi": 38, "is_liquid": True, "category": "juice"},
    "carrot juice": {"gi": 43, "is_liquid": True, "category": "juice"},
    "vegetable juice": {"gi": 43, "is_liquid": True, "category": "juice"},
    "juice box": {"gi": 65, "is_liquid": True, "category": "juice"},
    "fruit punch": {"gi": 67, "is_liquid": True, "category": "juice"},

    # Sodas & Sports Drinks
    "coca cola": {"gi": 63, "is_liquid": True, "category": "soda"},
    "coke": {"gi": 63, "is_liquid": True, "category": "soda"},
    "pepsi": {"gi": 58, "is_liquid": True, "category": "soda"},
    "sprite": {"gi": 59, "is_liquid": True, "category": "soda"},
    "7up": {"gi": 58, "is_liquid": True, "category": "soda"},
    "fanta": {"gi": 68, "is_liquid": True, "category": "soda"},
    "mountain dew": {"gi": 55, "is_liquid": True, "category": "soda"},
    "root beer": {"gi": 26, "is_liquid": True, "category": "soda"},
    "ginger ale": {"gi": 54, "is_liquid": True, "category": "soda"},
    "soda": {"gi": 60, "is_liquid": True, "category": "soda"},
    "gatorade": {"gi": 89, "is_liquid": True, "category": "sports_drink"},
    "powerade": {"gi": 80, "is_liquid": True, "category": "sports_drink"},
    "sports drink": {"gi": 85, "is_liquid": True, "category": "sports_drink"},
    "energy drink": {"gi": 70, "is_liquid": True, "category": "sports_drink"},
    "red bull": {"gi": 70, "is_liquid": True, "category": "sports_drink"},

    # Milk & Milk Drinks
    "milk": {"gi": 39, "is_liquid": True, "category": "dairy"},
    "whole milk": {"gi": 39, "is_liquid": True, "category": "dairy"},
    "skim milk": {"gi": 32, "is_liquid": True, "category": "dairy"},
    "2% milk": {"gi": 34, "is_liquid": True, "category": "dairy"},
    "chocolate milk": {"gi": 84, "is_liquid": True, "category": "dairy"},
    "choc milk": {"gi": 84, "is_liquid": True, "category": "dairy"},
    "strawberry milk": {"gi": 82, "is_liquid": True, "category": "dairy"},
    "nesquik": {"gi": 41, "is_liquid": True, "category": "dairy"},
    "milkshake": {"gi": 75, "is_liquid": True, "category": "dairy"},
    "vanilla shake": {"gi": 51, "is_liquid": True, "category": "dairy"},
    "chocolate shake": {"gi": 55, "is_liquid": True, "category": "dairy"},
    "smoothie": {"gi": 68, "is_liquid": True, "category": "smoothie"},
    "fruit smoothie": {"gi": 68, "is_liquid": True, "category": "smoothie"},
    "protein shake": {"gi": 35, "is_liquid": True, "category": "smoothie"},
    "ensure": {"gi": 50, "is_liquid": True, "category": "nutrition_drink"},
    "boost": {"gi": 53, "is_liquid": True, "category": "nutrition_drink"},
    "pedialyte": {"gi": 45, "is_liquid": True, "category": "nutrition_drink"},
    "oat milk": {"gi": 69, "is_liquid": True, "category": "dairy_alt"},
    "almond milk": {"gi": 30, "is_liquid": True, "category": "dairy_alt"},
    "soy milk": {"gi": 34, "is_liquid": True, "category": "dairy_alt"},

    # Glucose Treatment (for lows)
    "glucose tabs": {"gi": 100, "is_liquid": False, "category": "treatment"},
    "glucose tablets": {"gi": 100, "is_liquid": False, "category": "treatment"},
    "dex tabs": {"gi": 100, "is_liquid": False, "category": "treatment"},
    "dextrose": {"gi": 100, "is_liquid": False, "category": "treatment"},
    "glucose": {"gi": 100, "is_liquid": False, "category": "treatment"},
    "glucose gel": {"gi": 95, "is_liquid": True, "category": "treatment"},
    "insta glucose": {"gi": 95, "is_liquid": True, "category": "treatment"},
    "honey": {"gi": 61, "is_liquid": False, "category": "treatment"},

    # =====================================================
    # BREADS & BAKERY
    # =====================================================
    "white bread": {"gi": 75, "is_liquid": False, "category": "bread"},
    "wonder bread": {"gi": 75, "is_liquid": False, "category": "bread"},
    "wheat bread": {"gi": 69, "is_liquid": False, "category": "bread"},
    "whole wheat bread": {"gi": 69, "is_liquid": False, "category": "bread"},
    "whole grain bread": {"gi": 45, "is_liquid": False, "category": "bread"},
    "sourdough bread": {"gi": 54, "is_liquid": False, "category": "bread"},
    "sourdough": {"gi": 54, "is_liquid": False, "category": "bread"},
    "rye bread": {"gi": 58, "is_liquid": False, "category": "bread"},
    "pumpernickel": {"gi": 50, "is_liquid": False, "category": "bread"},
    "pita bread": {"gi": 68, "is_liquid": False, "category": "bread"},
    "pita": {"gi": 68, "is_liquid": False, "category": "bread"},
    "bagel": {"gi": 72, "is_liquid": False, "category": "bread"},
    "plain bagel": {"gi": 72, "is_liquid": False, "category": "bread"},
    "english muffin": {"gi": 77, "is_liquid": False, "category": "bread"},
    "croissant": {"gi": 67, "is_liquid": False, "category": "bread"},
    "hamburger bun": {"gi": 61, "is_liquid": False, "category": "bread"},
    "hot dog bun": {"gi": 61, "is_liquid": False, "category": "bread"},
    "biscuit": {"gi": 69, "is_liquid": False, "category": "bread"},
    "cornbread": {"gi": 110, "is_liquid": False, "category": "bread"},
    "bread": {"gi": 70, "is_liquid": False, "category": "bread"},
    "toast": {"gi": 70, "is_liquid": False, "category": "bread"},

    # Tortillas & Wraps
    "flour tortilla": {"gi": 30, "is_liquid": False, "category": "bread"},
    "corn tortilla": {"gi": 52, "is_liquid": False, "category": "bread"},
    "tortilla": {"gi": 30, "is_liquid": False, "category": "bread"},
    "wrap": {"gi": 35, "is_liquid": False, "category": "bread"},

    # =====================================================
    # BREAKFAST CEREALS
    # =====================================================
    "rice krispies": {"gi": 82, "is_liquid": False, "category": "cereal"},
    "corn flakes": {"gi": 81, "is_liquid": False, "category": "cereal"},
    "cheerios": {"gi": 74, "is_liquid": False, "category": "cereal"},
    "frosted flakes": {"gi": 55, "is_liquid": False, "category": "cereal"},
    "froot loops": {"gi": 69, "is_liquid": False, "category": "cereal"},
    "fruit loops": {"gi": 69, "is_liquid": False, "category": "cereal"},
    "lucky charms": {"gi": 55, "is_liquid": False, "category": "cereal"},
    "honey nut cheerios": {"gi": 74, "is_liquid": False, "category": "cereal"},
    "special k": {"gi": 69, "is_liquid": False, "category": "cereal"},
    "raisin bran": {"gi": 61, "is_liquid": False, "category": "cereal"},
    "all bran": {"gi": 42, "is_liquid": False, "category": "cereal"},
    "bran flakes": {"gi": 74, "is_liquid": False, "category": "cereal"},
    "grape nuts": {"gi": 75, "is_liquid": False, "category": "cereal"},
    "life cereal": {"gi": 66, "is_liquid": False, "category": "cereal"},
    "cocoa puffs": {"gi": 77, "is_liquid": False, "category": "cereal"},
    "cinnamon toast crunch": {"gi": 75, "is_liquid": False, "category": "cereal"},
    "cereal": {"gi": 70, "is_liquid": False, "category": "cereal"},

    # Oatmeal
    "oatmeal": {"gi": 55, "is_liquid": False, "category": "cereal"},
    "instant oatmeal": {"gi": 66, "is_liquid": False, "category": "cereal"},
    "steel cut oats": {"gi": 52, "is_liquid": False, "category": "cereal"},
    "rolled oats": {"gi": 55, "is_liquid": False, "category": "cereal"},
    "granola": {"gi": 55, "is_liquid": False, "category": "cereal"},
    "muesli": {"gi": 57, "is_liquid": False, "category": "cereal"},

    # Pancakes & Waffles
    "pancakes": {"gi": 66, "is_liquid": False, "category": "breakfast"},
    "pancake": {"gi": 66, "is_liquid": False, "category": "breakfast"},
    "waffles": {"gi": 76, "is_liquid": False, "category": "breakfast"},
    "waffle": {"gi": 76, "is_liquid": False, "category": "breakfast"},
    "french toast": {"gi": 59, "is_liquid": False, "category": "breakfast"},

    # =====================================================
    # GRAINS & STARCHES
    # =====================================================
    "white rice": {"gi": 73, "is_liquid": False, "category": "grain"},
    "jasmine rice": {"gi": 109, "is_liquid": False, "category": "grain"},
    "basmati rice": {"gi": 58, "is_liquid": False, "category": "grain"},
    "brown rice": {"gi": 50, "is_liquid": False, "category": "grain"},
    "wild rice": {"gi": 57, "is_liquid": False, "category": "grain"},
    "instant rice": {"gi": 87, "is_liquid": False, "category": "grain"},
    "rice": {"gi": 65, "is_liquid": False, "category": "grain"},
    "fried rice": {"gi": 65, "is_liquid": False, "category": "grain"},

    # Pasta
    "spaghetti": {"gi": 49, "is_liquid": False, "category": "pasta"},
    "white spaghetti": {"gi": 49, "is_liquid": False, "category": "pasta"},
    "whole wheat spaghetti": {"gi": 42, "is_liquid": False, "category": "pasta"},
    "linguine": {"gi": 52, "is_liquid": False, "category": "pasta"},
    "fettuccine": {"gi": 40, "is_liquid": False, "category": "pasta"},
    "macaroni": {"gi": 47, "is_liquid": False, "category": "pasta"},
    "penne": {"gi": 50, "is_liquid": False, "category": "pasta"},
    "rigatoni": {"gi": 45, "is_liquid": False, "category": "pasta"},
    "pasta": {"gi": 50, "is_liquid": False, "category": "pasta"},
    "noodles": {"gi": 47, "is_liquid": False, "category": "pasta"},
    "ramen": {"gi": 52, "is_liquid": False, "category": "pasta"},
    "egg noodles": {"gi": 40, "is_liquid": False, "category": "pasta"},
    "mac and cheese": {"gi": 64, "is_liquid": False, "category": "pasta"},
    "kraft mac and cheese": {"gi": 64, "is_liquid": False, "category": "pasta"},
    "mac n cheese": {"gi": 64, "is_liquid": False, "category": "pasta"},
    "lasagna": {"gi": 50, "is_liquid": False, "category": "pasta"},

    # Potatoes
    "baked potato": {"gi": 85, "is_liquid": False, "category": "potato"},
    "boiled potato": {"gi": 78, "is_liquid": False, "category": "potato"},
    "mashed potatoes": {"gi": 83, "is_liquid": False, "category": "potato"},
    "mashed potato": {"gi": 83, "is_liquid": False, "category": "potato"},
    "french fries": {"gi": 75, "is_liquid": False, "category": "potato"},
    "fries": {"gi": 75, "is_liquid": False, "category": "potato"},
    "hash browns": {"gi": 75, "is_liquid": False, "category": "potato"},
    "tater tots": {"gi": 75, "is_liquid": False, "category": "potato"},
    "potato chips": {"gi": 56, "is_liquid": False, "category": "snack"},
    "chips": {"gi": 56, "is_liquid": False, "category": "snack"},
    "potato": {"gi": 78, "is_liquid": False, "category": "potato"},
    "potatoes": {"gi": 78, "is_liquid": False, "category": "potato"},
    "sweet potato": {"gi": 63, "is_liquid": False, "category": "potato"},
    "yam": {"gi": 51, "is_liquid": False, "category": "potato"},

    # =====================================================
    # FRUITS
    # =====================================================
    "apple": {"gi": 36, "is_liquid": False, "category": "fruit"},
    "banana": {"gi": 51, "is_liquid": False, "category": "fruit"},
    "ripe banana": {"gi": 62, "is_liquid": False, "category": "fruit"},
    "green banana": {"gi": 42, "is_liquid": False, "category": "fruit"},
    "orange": {"gi": 43, "is_liquid": False, "category": "fruit"},
    "grapes": {"gi": 59, "is_liquid": False, "category": "fruit"},
    "watermelon": {"gi": 76, "is_liquid": False, "category": "fruit"},
    "cantaloupe": {"gi": 65, "is_liquid": False, "category": "fruit"},
    "honeydew": {"gi": 65, "is_liquid": False, "category": "fruit"},
    "pineapple": {"gi": 66, "is_liquid": False, "category": "fruit"},
    "mango": {"gi": 56, "is_liquid": False, "category": "fruit"},
    "papaya": {"gi": 56, "is_liquid": False, "category": "fruit"},
    "peach": {"gi": 42, "is_liquid": False, "category": "fruit"},
    "pear": {"gi": 38, "is_liquid": False, "category": "fruit"},
    "plum": {"gi": 39, "is_liquid": False, "category": "fruit"},
    "cherries": {"gi": 22, "is_liquid": False, "category": "fruit"},
    "strawberries": {"gi": 41, "is_liquid": False, "category": "fruit"},
    "blueberries": {"gi": 53, "is_liquid": False, "category": "fruit"},
    "raspberries": {"gi": 32, "is_liquid": False, "category": "fruit"},
    "blackberries": {"gi": 25, "is_liquid": False, "category": "fruit"},
    "kiwi": {"gi": 53, "is_liquid": False, "category": "fruit"},
    "grapefruit": {"gi": 25, "is_liquid": False, "category": "fruit"},
    "apricot": {"gi": 34, "is_liquid": False, "category": "fruit"},

    # Dried Fruits
    "raisins": {"gi": 64, "is_liquid": False, "category": "fruit"},
    "dates": {"gi": 42, "is_liquid": False, "category": "fruit"},
    "dried apricots": {"gi": 31, "is_liquid": False, "category": "fruit"},
    "dried cranberries": {"gi": 65, "is_liquid": False, "category": "fruit"},
    "prunes": {"gi": 29, "is_liquid": False, "category": "fruit"},
    "fruit snacks": {"gi": 78, "is_liquid": False, "category": "snack"},

    # =====================================================
    # CANDY & SWEETS
    # =====================================================
    "candy": {"gi": 80, "is_liquid": False, "category": "candy"},
    "gummy bears": {"gi": 78, "is_liquid": False, "category": "candy"},
    "gummies": {"gi": 78, "is_liquid": False, "category": "candy"},
    "skittles": {"gi": 70, "is_liquid": False, "category": "candy"},
    "m&ms": {"gi": 33, "is_liquid": False, "category": "candy"},
    "snickers": {"gi": 51, "is_liquid": False, "category": "candy"},
    "twix": {"gi": 44, "is_liquid": False, "category": "candy"},
    "kit kat": {"gi": 69, "is_liquid": False, "category": "candy"},
    "reeses": {"gi": 33, "is_liquid": False, "category": "candy"},
    "milky way": {"gi": 62, "is_liquid": False, "category": "candy"},
    "mars bar": {"gi": 65, "is_liquid": False, "category": "candy"},
    "jelly beans": {"gi": 78, "is_liquid": False, "category": "candy"},
    "licorice": {"gi": 78, "is_liquid": False, "category": "candy"},
    "lifesavers": {"gi": 70, "is_liquid": False, "category": "candy"},
    "starburst": {"gi": 70, "is_liquid": False, "category": "candy"},
    "nerds": {"gi": 85, "is_liquid": False, "category": "candy"},
    "pixie stix": {"gi": 100, "is_liquid": False, "category": "candy"},
    "lollipop": {"gi": 75, "is_liquid": False, "category": "candy"},

    # Chocolate
    "chocolate": {"gi": 40, "is_liquid": False, "category": "chocolate"},
    "milk chocolate": {"gi": 44, "is_liquid": False, "category": "chocolate"},
    "dark chocolate": {"gi": 23, "is_liquid": False, "category": "chocolate"},
    "white chocolate": {"gi": 44, "is_liquid": False, "category": "chocolate"},
    "chocolate bar": {"gi": 44, "is_liquid": False, "category": "chocolate"},
    "hershey bar": {"gi": 44, "is_liquid": False, "category": "chocolate"},

    # =====================================================
    # DESSERTS & BAKED GOODS
    # =====================================================
    "cake": {"gi": 67, "is_liquid": False, "category": "dessert"},
    "birthday cake": {"gi": 67, "is_liquid": False, "category": "dessert"},
    "chocolate cake": {"gi": 38, "is_liquid": False, "category": "dessert"},
    "vanilla cake": {"gi": 67, "is_liquid": False, "category": "dessert"},
    "angel food cake": {"gi": 67, "is_liquid": False, "category": "dessert"},
    "pound cake": {"gi": 54, "is_liquid": False, "category": "dessert"},
    "cheesecake": {"gi": 32, "is_liquid": False, "category": "dessert"},
    "cupcake": {"gi": 65, "is_liquid": False, "category": "dessert"},
    "muffin": {"gi": 60, "is_liquid": False, "category": "dessert"},
    "blueberry muffin": {"gi": 59, "is_liquid": False, "category": "dessert"},
    "donut": {"gi": 76, "is_liquid": False, "category": "dessert"},
    "doughnut": {"gi": 76, "is_liquid": False, "category": "dessert"},
    "brownie": {"gi": 72, "is_liquid": False, "category": "dessert"},
    "cookie": {"gi": 55, "is_liquid": False, "category": "dessert"},
    "cookies": {"gi": 55, "is_liquid": False, "category": "dessert"},
    "chocolate chip cookie": {"gi": 55, "is_liquid": False, "category": "dessert"},
    "oreo": {"gi": 57, "is_liquid": False, "category": "dessert"},
    "graham cracker": {"gi": 74, "is_liquid": False, "category": "dessert"},
    "pie": {"gi": 59, "is_liquid": False, "category": "dessert"},
    "apple pie": {"gi": 44, "is_liquid": False, "category": "dessert"},

    # Ice Cream
    "ice cream": {"gi": 51, "is_liquid": False, "category": "dessert"},
    "vanilla ice cream": {"gi": 62, "is_liquid": False, "category": "dessert"},
    "chocolate ice cream": {"gi": 57, "is_liquid": False, "category": "dessert"},
    "frozen yogurt": {"gi": 33, "is_liquid": False, "category": "dessert"},
    "popsicle": {"gi": 65, "is_liquid": False, "category": "dessert"},
    "ice cream cone": {"gi": 65, "is_liquid": False, "category": "dessert"},
    "sundae": {"gi": 55, "is_liquid": False, "category": "dessert"},

    # =====================================================
    # FAST FOOD & MEALS
    # =====================================================
    "pizza": {"gi": 60, "is_liquid": False, "category": "fast_food"},
    "pepperoni pizza": {"gi": 60, "is_liquid": False, "category": "fast_food"},
    "cheese pizza": {"gi": 60, "is_liquid": False, "category": "fast_food"},
    "pizza hut": {"gi": 60, "is_liquid": False, "category": "fast_food"},
    "dominos": {"gi": 60, "is_liquid": False, "category": "fast_food"},
    "hamburger": {"gi": 66, "is_liquid": False, "category": "fast_food"},
    "cheeseburger": {"gi": 66, "is_liquid": False, "category": "fast_food"},
    "mcdonalds": {"gi": 66, "is_liquid": False, "category": "fast_food"},
    "big mac": {"gi": 55, "is_liquid": False, "category": "fast_food"},
    "quarter pounder": {"gi": 55, "is_liquid": False, "category": "fast_food"},
    "whopper": {"gi": 55, "is_liquid": False, "category": "fast_food"},
    "burger king": {"gi": 55, "is_liquid": False, "category": "fast_food"},
    "hot dog": {"gi": 52, "is_liquid": False, "category": "fast_food"},
    "corn dog": {"gi": 52, "is_liquid": False, "category": "fast_food"},
    "chicken nuggets": {"gi": 46, "is_liquid": False, "category": "fast_food"},
    "nuggets": {"gi": 46, "is_liquid": False, "category": "fast_food"},
    "fish sticks": {"gi": 38, "is_liquid": False, "category": "fast_food"},
    "taco": {"gi": 68, "is_liquid": False, "category": "fast_food"},
    "tacos": {"gi": 68, "is_liquid": False, "category": "fast_food"},
    "taco bell": {"gi": 68, "is_liquid": False, "category": "fast_food"},
    "burrito": {"gi": 52, "is_liquid": False, "category": "fast_food"},
    "quesadilla": {"gi": 55, "is_liquid": False, "category": "fast_food"},
    "nachos": {"gi": 62, "is_liquid": False, "category": "fast_food"},
    "grilled cheese": {"gi": 55, "is_liquid": False, "category": "sandwich"},
    "pb&j": {"gi": 55, "is_liquid": False, "category": "sandwich"},
    "peanut butter jelly": {"gi": 55, "is_liquid": False, "category": "sandwich"},
    "sandwich": {"gi": 55, "is_liquid": False, "category": "sandwich"},
    "sub": {"gi": 52, "is_liquid": False, "category": "sandwich"},
    "subway": {"gi": 52, "is_liquid": False, "category": "sandwich"},

    # =====================================================
    # SNACKS
    # =====================================================
    "popcorn": {"gi": 65, "is_liquid": False, "category": "snack"},
    "pretzels": {"gi": 83, "is_liquid": False, "category": "snack"},
    "crackers": {"gi": 74, "is_liquid": False, "category": "snack"},
    "saltines": {"gi": 74, "is_liquid": False, "category": "snack"},
    "goldfish": {"gi": 55, "is_liquid": False, "category": "snack"},
    "cheez its": {"gi": 55, "is_liquid": False, "category": "snack"},
    "triscuits": {"gi": 67, "is_liquid": False, "category": "snack"},
    "wheat thins": {"gi": 67, "is_liquid": False, "category": "snack"},
    "rice cakes": {"gi": 87, "is_liquid": False, "category": "snack"},
    "granola bar": {"gi": 56, "is_liquid": False, "category": "snack"},
    "cliff bar": {"gi": 56, "is_liquid": False, "category": "snack"},
    "protein bar": {"gi": 35, "is_liquid": False, "category": "snack"},
    "tortilla chips": {"gi": 63, "is_liquid": False, "category": "snack"},
    "doritos": {"gi": 65, "is_liquid": False, "category": "snack"},
    "cheetos": {"gi": 65, "is_liquid": False, "category": "snack"},

    # =====================================================
    # DAIRY & PROTEIN
    # =====================================================
    "yogurt": {"gi": 41, "is_liquid": False, "category": "dairy"},
    "greek yogurt": {"gi": 12, "is_liquid": False, "category": "dairy"},
    "fruit yogurt": {"gi": 33, "is_liquid": False, "category": "dairy"},
    "cottage cheese": {"gi": 10, "is_liquid": False, "category": "dairy"},
    "cheese": {"gi": 0, "is_liquid": False, "category": "dairy"},
    "cream cheese": {"gi": 0, "is_liquid": False, "category": "dairy"},
    "ice cream": {"gi": 51, "is_liquid": False, "category": "dairy"},

    # =====================================================
    # LEGUMES (Low GI)
    # =====================================================
    "beans": {"gi": 32, "is_liquid": False, "category": "legume"},
    "black beans": {"gi": 30, "is_liquid": False, "category": "legume"},
    "kidney beans": {"gi": 24, "is_liquid": False, "category": "legume"},
    "pinto beans": {"gi": 39, "is_liquid": False, "category": "legume"},
    "navy beans": {"gi": 38, "is_liquid": False, "category": "legume"},
    "chickpeas": {"gi": 28, "is_liquid": False, "category": "legume"},
    "lentils": {"gi": 32, "is_liquid": False, "category": "legume"},
    "hummus": {"gi": 25, "is_liquid": False, "category": "legume"},
    "split peas": {"gi": 32, "is_liquid": False, "category": "legume"},
    "baked beans": {"gi": 40, "is_liquid": False, "category": "legume"},

    # =====================================================
    # NUTS & SEEDS (Very Low GI)
    # =====================================================
    "peanuts": {"gi": 14, "is_liquid": False, "category": "nuts"},
    "peanut butter": {"gi": 14, "is_liquid": False, "category": "nuts"},
    "almonds": {"gi": 15, "is_liquid": False, "category": "nuts"},
    "cashews": {"gi": 22, "is_liquid": False, "category": "nuts"},
    "walnuts": {"gi": 15, "is_liquid": False, "category": "nuts"},
    "pecans": {"gi": 10, "is_liquid": False, "category": "nuts"},
    "mixed nuts": {"gi": 15, "is_liquid": False, "category": "nuts"},
    "sunflower seeds": {"gi": 35, "is_liquid": False, "category": "nuts"},
    "trail mix": {"gi": 45, "is_liquid": False, "category": "nuts"},

    # =====================================================
    # VEGETABLES (Low GI)
    # =====================================================
    "corn": {"gi": 52, "is_liquid": False, "category": "vegetable"},
    "peas": {"gi": 48, "is_liquid": False, "category": "vegetable"},
    "carrots": {"gi": 47, "is_liquid": False, "category": "vegetable"},
    "broccoli": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "green beans": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "tomato": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "lettuce": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "spinach": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "celery": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "cucumber": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "bell pepper": {"gi": 15, "is_liquid": False, "category": "vegetable"},
    "mushrooms": {"gi": 15, "is_liquid": False, "category": "vegetable"},
}


def lookup_gi(food_name: str) -> dict:
    """
    Look up glycemic index for a food.

    Returns: dict with {gi, is_liquid, category, source} or None if not found.
    """
    food_lower = food_name.lower().strip()

    # Direct match
    if food_lower in GLYCEMIC_INDEX_DATABASE:
        result = GLYCEMIC_INDEX_DATABASE[food_lower].copy()
        result["source"] = "database"
        result["matched"] = food_lower
        return result

    # Partial match - find foods that contain the search term
    for db_food, props in GLYCEMIC_INDEX_DATABASE.items():
        if db_food in food_lower or food_lower in db_food:
            result = props.copy()
            result["source"] = "database_partial"
            result["matched"] = db_food
            return result

    # Not found
    return None


def get_all_foods() -> list:
    """Return list of all food names in database."""
    return list(GLYCEMIC_INDEX_DATABASE.keys())


def get_foods_by_category(category: str) -> list:
    """Return list of foods in a specific category."""
    return [
        name for name, props in GLYCEMIC_INDEX_DATABASE.items()
        if props.get("category") == category
    ]


def get_categories() -> list:
    """Return list of all unique categories."""
    return list(set(props.get("category") for props in GLYCEMIC_INDEX_DATABASE.values()))
