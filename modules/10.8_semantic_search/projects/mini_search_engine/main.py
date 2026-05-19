# -*- coding: utf-8 -*-
# main.py
#
# Module 10.8 -- Semantic Search Systems
# Project: Mini Search Engine
#
# HOW TO RUN:
#   cd modules/10.8_semantic_search
#   python projects/mini_search_engine/main.py
#
# WHAT THIS DOES:
#   1. On startup: indexes all 50 Wikipedia article summaries
#   2. Enters an interactive loop: user types a query
#   3. For each query: bi-encoder retrieves top-20 candidates
#                      cross-encoder reranks to top-5
#                      results displayed with title + snippet
#   4. Type 'quit' or 'exit' to stop

import sys
import os

# Add the project directory to Python's module search path
# This lets us import from indexer.py, searcher.py, reranker.py in the same folder
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Import our project modules
from indexer import Indexer      # Loads and encodes documents
from searcher import Searcher    # Bi-encoder style retrieval
from reranker import Reranker    # Cross-encoder style re-ranking


# ============================================================
# THE 50 WIKIPEDIA ARTICLE SUMMARIES
# ============================================================
# Organized by topic: Science (10), History (10), Technology (10),
# Sports (10), Geography (10)
# These are SUMMARIES, not full articles. No internet needed.

WIKIPEDIA_ARTICLES = [

    # =====================
    # SCIENCE (10 articles)
    # =====================
    {
        "id": "sci_gravity",
        "title": "Gravity",
        "text": ("Gravity is a natural phenomenon by which all things with mass or energy "
                 "are attracted toward one another. On Earth, gravity gives weight to "
                 "physical objects. The Moon's gravity causes the tides of the oceans. "
                 "Gravity also holds Earth and the planets in their orbits around the Sun. "
                 "Isaac Newton formulated the law of universal gravitation in 1687. "
                 "Albert Einstein later described gravity as a curvature of spacetime "
                 "in his theory of general relativity.")
    },
    {
        "id": "sci_blackhole",
        "title": "Black hole",
        "text": ("A black hole is a region of spacetime where gravity is so strong that "
                 "nothing, not even light or other electromagnetic waves, can escape. "
                 "The boundary beyond which escape is impossible is called the event horizon. "
                 "Black holes are formed when massive stars collapse at the end of their lives. "
                 "They can also form from the collision of two neutron stars. "
                 "Supermassive black holes exist at the center of most galaxies. "
                 "The first image of a black hole was captured in 2019 by the Event Horizon Telescope.")
    },
    {
        "id": "sci_dna",
        "title": "DNA",
        "text": ("Deoxyribonucleic acid (DNA) is a polymer composed of two chains that coil "
                 "around each other to form a double helix. DNA carries the genetic instructions "
                 "for the development, functioning, growth and reproduction of all known organisms. "
                 "It was first identified in 1869 by Friedrich Miescher. "
                 "The structure of DNA was described by James Watson and Francis Crick in 1953. "
                 "Each human cell contains about 3 billion base pairs of DNA. "
                 "DNA mutations can cause cancer and genetic diseases.")
    },
    {
        "id": "sci_evolution",
        "title": "Evolution",
        "text": ("Evolution is the change in heritable characteristics of biological populations "
                 "over successive generations. It occurs through natural selection, mutation, "
                 "genetic drift, and gene flow. Charles Darwin published the theory of evolution "
                 "by natural selection in On the Origin of Species in 1859. "
                 "Evolution explains the biodiversity of life on Earth. "
                 "All life shares common ancestors. "
                 "The fossil record provides evidence of evolutionary change over billions of years.")
    },
    {
        "id": "sci_photosynthesis",
        "title": "Photosynthesis",
        "text": ("Photosynthesis is a process used by plants, algae and cyanobacteria to convert "
                 "light energy into chemical energy stored in glucose or other sugars. "
                 "It takes in carbon dioxide and water and releases oxygen as a byproduct. "
                 "Chlorophyll is the green pigment that absorbs sunlight in plant leaves. "
                 "Photosynthesis produces the oxygen in Earth's atmosphere. "
                 "It is the foundation of most food chains on Earth. "
                 "Approximately 100 billion tonnes of carbon are fixed by photosynthesis per year.")
    },
    {
        "id": "sci_relativity",
        "title": "General relativity",
        "text": ("General relativity is Albert Einstein's theory of gravitation published in 1915. "
                 "It generalizes special relativity and Newton's law of gravitation. "
                 "In general relativity, gravity is described as the curvature of spacetime "
                 "caused by mass and energy. It predicts phenomena such as gravitational waves, "
                 "the bending of light around massive objects, and the expansion of the universe. "
                 "It has been confirmed by many experiments including GPS corrections, "
                 "observations of binary pulsars, and gravitational wave detection by LIGO.")
    },
    {
        "id": "sci_quantum",
        "title": "Quantum mechanics",
        "text": ("Quantum mechanics is a fundamental theory of physics that describes the behavior "
                 "of matter and energy at the smallest scales of atoms and subatomic particles. "
                 "It introduces concepts like wave-particle duality, the uncertainty principle, "
                 "and quantum superposition. Max Planck and Albert Einstein were pioneers of "
                 "quantum theory in the early 1900s. "
                 "Quantum mechanics underlies modern electronics, lasers, and MRI machines. "
                 "Quantum computers exploit quantum phenomena to solve problems faster than "
                 "classical computers.")
    },
    {
        "id": "sci_climate",
        "title": "Climate change",
        "text": ("Climate change refers to long-term shifts in global temperatures and weather patterns. "
                 "Since the Industrial Revolution, human activities, especially burning fossil fuels, "
                 "have been the main driver of climate change. "
                 "Rising carbon dioxide levels cause the greenhouse effect to intensify. "
                 "Effects include rising sea levels, more extreme weather events, "
                 "melting glaciers and polar ice, and shifts in plant and animal habitats. "
                 "The Paris Agreement aims to limit global temperature rise to 1.5 degrees Celsius. "
                 "Renewable energy and reforestation are key solutions.")
    },
    {
        "id": "sci_ecosystem",
        "title": "Ecosystem",
        "text": ("An ecosystem is a geographic area where plants, animals, and other organisms, "
                 "along with weather and landscape, work together to form a bubble of life. "
                 "Ecosystems contain biotic elements (living things) and abiotic elements "
                 "(non-living things like rocks, temperature, and water). "
                 "Examples include rainforests, coral reefs, deserts, and tundras. "
                 "Ecosystems provide services like clean water, food, and climate regulation. "
                 "Human activities like deforestation and pollution damage ecosystems. "
                 "Biodiversity is a key measure of ecosystem health.")
    },
    {
        "id": "sci_universe",
        "title": "Universe",
        "text": ("The universe is all of space, time, matter, and energy. "
                 "It began with the Big Bang approximately 13.8 billion years ago. "
                 "The observable universe is about 93 billion light-years in diameter. "
                 "It contains at least two trillion galaxies. "
                 "Dark matter and dark energy make up about 95 percent of the universe. "
                 "The universe is expanding, and this expansion is accelerating. "
                 "The fate of the universe depends on the amount of dark energy.")
    },

    # ========================
    # HISTORY (10 articles)
    # ========================
    {
        "id": "hist_ww2",
        "title": "World War II",
        "text": ("World War II was a global war that lasted from 1939 to 1945. "
                 "It involved most of the world's nations and was the deadliest conflict "
                 "in history, killing an estimated 70-85 million people. "
                 "The Allied Powers (USA, UK, Soviet Union, France) defeated the Axis Powers "
                 "(Germany, Japan, Italy). "
                 "The Holocaust was the genocide of six million Jews and millions of others. "
                 "The war ended with the atomic bombings of Hiroshima and Nagasaki. "
                 "It led to the formation of the United Nations and the Cold War.")
    },
    {
        "id": "hist_roman",
        "title": "Roman Empire",
        "text": ("The Roman Empire was the post-Republican state of ancient Rome. "
                 "At its height under Emperor Trajan in 117 AD, it covered 5 million km2. "
                 "It began when Augustus Caesar became the first Roman Emperor in 27 BC. "
                 "The Roman Empire is known for its law, architecture, and language (Latin). "
                 "It split into the Western and Eastern Roman Empires in 285 AD. "
                 "The Western Roman Empire fell in 476 AD. "
                 "The Eastern Roman Empire (Byzantine Empire) continued until 1453 AD.")
    },
    {
        "id": "hist_egypt",
        "title": "Ancient Egypt",
        "text": ("Ancient Egypt was a civilization in northeastern Africa along the Nile River. "
                 "It lasted for over 3,000 years, from around 3100 BC to 30 BC. "
                 "Ancient Egyptians built the pyramids, the Sphinx, and elaborate temples. "
                 "They developed one of the world's earliest writing systems: hieroglyphics. "
                 "Pharaohs were the rulers of Ancient Egypt and were considered gods on Earth. "
                 "The civilization was united by Narmer around 3100 BC. "
                 "It was conquered by Alexander the Great in 332 BC.")
    },
    {
        "id": "hist_silk_road",
        "title": "Silk Road",
        "text": ("The Silk Road was a network of trade routes connecting China and the Far East "
                 "with the Middle East and Europe. It was named for the lucrative trade in silk "
                 "carried out along its length. Active from the 2nd century BC to the 15th century. "
                 "It also facilitated the spread of religions, art, technology, and disease. "
                 "The bubonic plague that caused the Black Death spread along the Silk Road. "
                 "Marco Polo traveled the Silk Road in the 13th century. "
                 "The modern Belt and Road Initiative is sometimes called the New Silk Road.")
    },
    {
        "id": "hist_french_rev",
        "title": "French Revolution",
        "text": ("The French Revolution was a period of radical political and societal change in France "
                 "that began in 1789 and ended in the late 1790s with Napoleon Bonaparte's rise. "
                 "It abolished the monarchy, established a republic, and caused upheaval through "
                 "Europe. Key causes were financial crises, social inequality, and Enlightenment ideas. "
                 "The Declaration of the Rights of Man was adopted in 1789. "
                 "The Reign of Terror saw mass executions including King Louis XVI and Marie Antoinette. "
                 "It shaped modern democracy and inspired revolutions worldwide.")
    },
    {
        "id": "hist_mongol",
        "title": "Mongol Empire",
        "text": ("The Mongol Empire was the largest contiguous empire in history, covering "
                 "24 million km2 at its height. It was founded by Genghis Khan in 1206. "
                 "The empire stretched from the Pacific Ocean to Eastern Europe. "
                 "The Mongols created the Pax Mongolica, a period of relative peace and "
                 "increased trade along the Silk Road. "
                 "Kublai Khan founded the Yuan dynasty in China. "
                 "The empire fragmented after Kublai Khan's death. "
                 "Mongol invasions killed millions and devastated many regions.")
    },
    {
        "id": "hist_renaissance",
        "title": "Renaissance",
        "text": ("The Renaissance was a period of European cultural, artistic, and scientific "
                 "rebirth that began in Italy in the 14th century. "
                 "It produced artists like Leonardo da Vinci and Michelangelo, "
                 "writers like Dante and Petrarch, and thinkers like Erasmus. "
                 "The printing press invented by Gutenberg accelerated the spread of Renaissance ideas. "
                 "It marked the transition from the medieval period to modernity. "
                 "Scientific advances by Copernicus and Galileo challenged church doctrine. "
                 "The Renaissance ended in the early 17th century.")
    },
    {
        "id": "hist_cold_war",
        "title": "Cold War",
        "text": ("The Cold War was a period of geopolitical tension between the United States "
                 "and the Soviet Union and their respective allies. "
                 "It lasted from approximately 1947 to 1991. "
                 "It was characterized by nuclear arms races, proxy wars, and the Space Race. "
                 "The Berlin Wall, built in 1961 and torn down in 1989, was its symbol. "
                 "The Cuban Missile Crisis of 1962 brought the world close to nuclear war. "
                 "It ended with the dissolution of the Soviet Union in 1991.")
    },
    {
        "id": "hist_slave_trade",
        "title": "Atlantic slave trade",
        "text": ("The Atlantic slave trade was the transportation of enslaved African people "
                 "to the Americas between the 16th and 19th centuries. "
                 "An estimated 12.5 million Africans were transported as slaves. "
                 "Slaves were forced to work on sugar, cotton, and tobacco plantations. "
                 "The trade was central to the economies of European colonial powers. "
                 "The abolitionist movement grew in the late 18th and early 19th centuries. "
                 "Britain abolished the slave trade in 1807. The USA did so in 1865.")
    },
    {
        "id": "hist_black_death",
        "title": "Black Death",
        "text": ("The Black Death was a bubonic plague pandemic that devastated Europe and Asia "
                 "in the 14th century. It killed 30-60 percent of Europe's population. "
                 "It was caused by the bacterium Yersinia pestis, spread by fleas on rats. "
                 "It arrived in Europe via Crimean trade ships in 1347. "
                 "It profoundly changed European society, culture, and religion. "
                 "The shortage of labor led to improved wages and conditions for peasants. "
                 "Recurring outbreaks continued until the 19th century.")
    },

    # ===========================
    # TECHNOLOGY (10 articles)
    # ===========================
    {
        "id": "tech_internet",
        "title": "Internet",
        "text": ("The Internet is a global system of interconnected computer networks. "
                 "It was developed from the ARPANET in the late 1960s by the US Department of Defense. "
                 "Tim Berners-Lee invented the World Wide Web in 1989, making the internet accessible "
                 "to the public. Today over 5 billion people use the internet. "
                 "It enables communication, commerce, education, and entertainment worldwide. "
                 "Social media, email, streaming, and e-commerce are major internet applications. "
                 "The internet uses protocols like TCP/IP and HTTP to transmit data.")
    },
    {
        "id": "tech_ai",
        "title": "Artificial intelligence",
        "text": ("Artificial intelligence (AI) is the simulation of human intelligence in machines. "
                 "It includes tasks like learning, reasoning, problem-solving, and language understanding. "
                 "AI was founded as a field in 1956 at the Dartmouth Workshop. "
                 "Machine learning is a subset of AI where systems learn from data. "
                 "Deep learning uses neural networks with many layers. "
                 "Modern AI powers search engines, virtual assistants, self-driving cars, "
                 "and medical diagnosis systems. GPT and BERT are large language models.")
    },
    {
        "id": "tech_computer",
        "title": "Computer",
        "text": ("A computer is a machine that can be programmed to carry out sequences of operations. "
                 "Modern computers are digital and use binary code. "
                 "Charles Babbage designed the first mechanical computer in the 19th century. "
                 "ENIAC, built in 1945, was one of the first electronic computers. "
                 "Transistors replaced vacuum tubes, making computers smaller. "
                 "Integrated circuits led to microprocessors and personal computers. "
                 "Today computers are found in smartphones, cars, appliances, and medical devices.")
    },
    {
        "id": "tech_blockchain",
        "title": "Blockchain",
        "text": ("Blockchain is a distributed ledger technology that records transactions "
                 "across many computers so that no single record can be altered. "
                 "Bitcoin, created in 2009 by Satoshi Nakamoto, uses blockchain. "
                 "Each block contains a cryptographic hash of the previous block. "
                 "Blockchain is decentralized and requires no central authority. "
                 "Smart contracts are programs stored on a blockchain. "
                 "Ethereum is a blockchain platform for decentralized applications. "
                 "Blockchain has applications in finance, supply chains, and healthcare.")
    },
    {
        "id": "tech_electricity",
        "title": "Electricity",
        "text": ("Electricity is the set of physical phenomena associated with electric charge. "
                 "Benjamin Franklin's experiments with lightning in the 1750s advanced understanding. "
                 "Thomas Edison invented the light bulb in 1879 and created the first power grid. "
                 "Nikola Tesla developed alternating current (AC) electrical systems. "
                 "Electricity is generated by power plants using steam turbines, wind, or solar panels. "
                 "It powers lights, computers, motors, and communication devices. "
                 "Global electricity consumption exceeds 25 thousand terawatt-hours per year.")
    },
    {
        "id": "tech_robot",
        "title": "Robotics",
        "text": ("Robotics is an interdisciplinary branch of computer science and engineering "
                 "that involves the design, construction, and use of robots. "
                 "Industrial robots have been used in manufacturing since the 1960s. "
                 "Modern robots use sensors, AI, and computer vision to navigate and interact. "
                 "Robots are used in surgery, exploration, agriculture, and logistics. "
                 "Boston Dynamics creates advanced humanoid and quadruped robots. "
                 "Autonomous robots include self-driving cars and delivery drones. "
                 "Concerns about robots replacing human jobs are a major societal discussion.")
    },
    {
        "id": "tech_satellite",
        "title": "Satellite",
        "text": ("A satellite is an object that orbits a larger object in space. "
                 "Sputnik, launched by the Soviet Union in 1957, was the first artificial satellite. "
                 "GPS satellites provide global navigation and positioning services. "
                 "Communications satellites relay phone calls, television, and internet signals. "
                 "Weather satellites monitor Earth's atmosphere and climate. "
                 "Earth observation satellites track deforestation, agriculture, and disasters. "
                 "SpaceX's Starlink is building a constellation of thousands of satellites "
                 "to provide global broadband internet.")
    },
    {
        "id": "tech_programming",
        "title": "Computer programming",
        "text": ("Computer programming is the process of writing instructions for a computer to execute. "
                 "Programs are written in programming languages like Python, Java, C, and JavaScript. "
                 "Assembly language was the first widely used programming language. "
                 "High-level languages abstract hardware details and improve productivity. "
                 "Software development involves design, coding, testing, and maintenance. "
                 "Algorithms are step-by-step procedures for solving problems. "
                 "Open source software allows anyone to read, modify, and distribute code. "
                 "GitHub is the largest code hosting platform in the world.")
    },
    {
        "id": "tech_semiconductor",
        "title": "Semiconductor",
        "text": ("A semiconductor is a material that has electrical conductivity between "
                 "that of a conductor (like copper) and an insulator (like glass). "
                 "Silicon is the most common semiconductor material. "
                 "Transistors, made from semiconductors, are the building blocks of modern electronics. "
                 "Intel, TSMC, and Samsung are major semiconductor manufacturers. "
                 "Moore's Law predicts transistor density doubles every two years. "
                 "Semiconductors are found in computers, smartphones, and solar panels. "
                 "Global chip shortages in 2021-2022 highlighted semiconductor supply chain risks.")
    },
    {
        "id": "tech_gps",
        "title": "GPS",
        "text": ("The Global Positioning System (GPS) is a satellite-based navigation system "
                 "owned by the United States government. "
                 "It was fully operational by 1995 and opened to civilian use. "
                 "GPS works by measuring the time it takes for signals from multiple satellites "
                 "to reach a receiver on Earth. "
                 "It provides accurate location, velocity, and time information anywhere on Earth. "
                 "GPS is used in navigation, mapping, aviation, shipping, and emergency services. "
                 "Smartphones use GPS for maps and location services.")
    },

    # ====================
    # SPORTS (10 articles)
    # ====================
    {
        "id": "sport_football",
        "title": "Association football",
        "text": ("Association football (soccer) is the world's most popular sport with 4 billion fans. "
                 "It is played between two teams of 11 players on a rectangular field. "
                 "The object is to score by kicking the ball into the opposing goal. "
                 "FIFA, founded in 1904, governs the sport internationally. "
                 "The FIFA World Cup, held every four years, is the most watched sporting event. "
                 "Brazil has won the most World Cups with five titles. "
                 "Lionel Messi and Cristiano Ronaldo are considered the greatest players of their era.")
    },
    {
        "id": "sport_tennis",
        "title": "Tennis",
        "text": ("Tennis is a racket sport played on a rectangular court. "
                 "It can be played by two players (singles) or four players (doubles). "
                 "The four Grand Slam tournaments are Wimbledon, the US Open, "
                 "the French Open, and the Australian Open. "
                 "Roger Federer, Rafael Nadal, and Novak Djokovic are the dominant players "
                 "of the modern era, each with over 20 Grand Slam titles. "
                 "Serena Williams is considered the greatest women's tennis player. "
                 "Tennis originated in 19th century England.")
    },
    {
        "id": "sport_olympics",
        "title": "Olympic Games",
        "text": ("The Olympic Games are the world's leading international sports competition. "
                 "They are held every four years, alternating between Summer and Winter Olympics. "
                 "The modern Olympics were founded by Pierre de Coubertin in 1896 in Athens. "
                 "The Olympics include over 300 events across 30+ sports. "
                 "The Olympic motto is Citius, Altius, Fortius (Faster, Higher, Stronger). "
                 "The most decorated Olympian is Michael Phelps with 28 medals. "
                 "The 2024 Summer Olympics were held in Paris, France.")
    },
    {
        "id": "sport_basketball",
        "title": "Basketball",
        "text": ("Basketball is a sport played between two teams of five players. "
                 "It was invented by Dr. James Naismith in 1891 in Springfield, Massachusetts. "
                 "The NBA (National Basketball Association) is the premier professional league. "
                 "Michael Jordan, LeBron James, and Kareem Abdul-Jabbar are the greatest players. "
                 "A standard basketball court is 28 by 15 meters. "
                 "Three-point shots are awarded for baskets from beyond the arc. "
                 "Basketball was introduced as an Olympic sport in 1936.")
    },
    {
        "id": "sport_cricket",
        "title": "Cricket",
        "text": ("Cricket is a bat-and-ball game played between two teams of 11 players. "
                 "It is the second most popular sport in the world by viewership. "
                 "Major formats include Test cricket (5 days), One Day Internationals, and T20. "
                 "The ICC Cricket World Cup is the premier international tournament. "
                 "Cricket originated in southeastern England in the 16th century. "
                 "India, Australia, England, and the West Indies are traditional powers. "
                 "Sachin Tendulkar holds most international batting records.")
    },
    {
        "id": "sport_swimming",
        "title": "Swimming",
        "text": ("Swimming is an individual or team sport that requires the use of arms and legs "
                 "to move through water. "
                 "Major competitive strokes are freestyle, backstroke, breaststroke, and butterfly. "
                 "Swimming has been an Olympic sport since the 1896 Athens Games. "
                 "Michael Phelps won 23 Olympic gold medals in swimming. "
                 "Competitive swimming pools are 50 meters long. "
                 "Open water swimming competitions are held in rivers, lakes, and oceans. "
                 "Swimming is also widely practiced as exercise and recreation.")
    },
    {
        "id": "sport_marathon",
        "title": "Marathon",
        "text": ("A marathon is a long-distance running race with an official distance of "
                 "42.195 kilometers. "
                 "It was introduced as an Olympic event in 1896, based on the legend of Pheidippides "
                 "running from the Battle of Marathon to Athens. "
                 "The Boston Marathon, held since 1897, is the world's oldest annual marathon. "
                 "Eliud Kipchoge broke the 2-hour marathon barrier in 2019 in Vienna. "
                 "Major marathons include London, Berlin, Chicago, New York, and Tokyo. "
                 "Elite marathon runners finish in just over 2 hours.")
    },
    {
        "id": "sport_chess",
        "title": "Chess",
        "text": ("Chess is a strategy board game for two players on an 8 by 8 board. "
                 "Each player controls 16 pieces: one king, one queen, two rooks, two knights, "
                 "two bishops, and eight pawns. "
                 "Chess originated in India around the 6th century AD. "
                 "FIDE governs international chess competition. "
                 "Magnus Carlsen of Norway became World Champion in 2013 and held the title for a decade. "
                 "IBM's Deep Blue defeated Garry Kasparov in 1997, the first time a computer "
                 "beat a world champion. "
                 "Chess is widely taught to children as a tool for intellectual development.")
    },
    {
        "id": "sport_cycling",
        "title": "Cycling",
        "text": ("Cycling is the use of bicycles for transport, recreation, exercise, or sport. "
                 "Competitive cycling includes road racing, track cycling, and mountain biking. "
                 "The Tour de France is the most prestigious cycling race, held annually since 1903. "
                 "It covers approximately 3,500 km over three weeks. "
                 "Eddie Merckx of Belgium won the Tour de France five times. "
                 "Lance Armstrong won seven consecutive Tours but was stripped of his titles. "
                 "Cycling became an Olympic sport in 1896.")
    },
    {
        "id": "sport_golf",
        "title": "Golf",
        "text": ("Golf is a sport in which players hit balls with clubs into a series of holes "
                 "on a course. The goal is to complete the course in as few strokes as possible. "
                 "The four major championships are The Masters, the US Open, The Open Championship, "
                 "and the PGA Championship. "
                 "Tiger Woods is considered the greatest golfer of his era with 15 major titles. "
                 "Golf originated in Scotland in the 15th century. "
                 "The Ryder Cup is a major team competition between Europe and the United States. "
                 "Golf has been an Olympic sport since 2016.")
    },

    # =======================
    # GEOGRAPHY (10 articles)
    # =======================
    {
        "id": "geo_amazon",
        "title": "Amazon River",
        "text": ("The Amazon River in South America is the largest river by discharge volume. "
                 "It flows through Brazil, Peru, and Colombia. "
                 "The Amazon is approximately 6,400 km long. "
                 "The Amazon basin contains the world's largest tropical rainforest. "
                 "It is home to over 10 percent of all species on Earth. "
                 "The Amazon produces 20 percent of the world's freshwater discharge to the ocean. "
                 "Deforestation threatens the Amazon ecosystem and global climate stability.")
    },
    {
        "id": "geo_everest",
        "title": "Mount Everest",
        "text": ("Mount Everest is Earth's highest mountain at 8,848.86 meters above sea level. "
                 "It is located in the Himalayas on the border between Nepal and Tibet. "
                 "Edmund Hillary and Tenzing Norgay were the first to summit it in 1953. "
                 "Over 5,000 people have successfully climbed Everest. "
                 "The mountain has two main routes: the Southeast Ridge from Nepal "
                 "and the North Ridge from Tibet. "
                 "Approximately 300 climbers have died on Everest. "
                 "The mountain is named after British surveyor George Everest.")
    },
    {
        "id": "geo_pacific",
        "title": "Pacific Ocean",
        "text": ("The Pacific Ocean is the largest and deepest ocean on Earth. "
                 "It covers more than 165 million km2, about one-third of Earth's surface. "
                 "The Mariana Trench in the Pacific is the deepest point on Earth "
                 "at 11,034 meters below sea level. "
                 "The Pacific is bordered by the Americas on the east and Asia and Australia on the west. "
                 "It contains over 25,000 islands. "
                 "The Pacific Ring of Fire contains most of the world's active volcanoes. "
                 "Great Pacific Garbage Patch is a massive marine debris accumulation zone.")
    },
    {
        "id": "geo_sahara",
        "title": "Sahara Desert",
        "text": ("The Sahara is the world's largest hot desert, covering 9.2 million km2 in North Africa. "
                 "It spans 11 countries including Algeria, Libya, Egypt, and Sudan. "
                 "Despite its harsh environment, the Sahara supports diverse wildlife including "
                 "fennec foxes, camels, and desert scorpions. "
                 "The ancient Sahara was green with rivers and lakes during the African Humid Period. "
                 "Tuareg nomads have inhabited the Sahara for thousands of years. "
                 "Temperatures can exceed 50 degrees Celsius in summer. "
                 "Sand dunes called ergs cover about 25 percent of the Sahara.")
    },
    {
        "id": "geo_nile",
        "title": "Nile River",
        "text": ("The Nile is a major north-flowing river in northeastern Africa. "
                 "With a length of about 6,650 km, it is one of the longest rivers in the world. "
                 "It flows through 11 countries including Ethiopia, Sudan, and Egypt. "
                 "The annual flooding of the Nile deposited rich soil and enabled Ancient Egypt "
                 "to become one of the great civilizations. "
                 "The Nile has two main tributaries: the Blue Nile and the White Nile. "
                 "Lake Victoria in East Africa is the source of the White Nile. "
                 "The Aswan High Dam was built in 1970 to control flooding and generate electricity.")
    },
    {
        "id": "geo_himalaya",
        "title": "Himalayas",
        "text": ("The Himalayas are a mountain range in South Asia, home to the world's highest peaks. "
                 "They stretch 2,400 km across Afghanistan, Pakistan, India, Nepal, Bhutan, and China. "
                 "The range contains over 110 peaks above 7,300 meters, including Mount Everest. "
                 "The Himalayas were formed by the collision of the Indian and Eurasian tectonic plates. "
                 "They are the source of many of Asia's great rivers including the Ganges and Yangtze. "
                 "The Himalayas act as a barrier to cold Arctic winds, giving South Asia its climate. "
                 "About 52 million people live in the Himalayan region.")
    },
    {
        "id": "geo_antarctica",
        "title": "Antarctica",
        "text": ("Antarctica is Earth's southernmost continent, containing the South Pole. "
                 "It is the coldest, driest, and windiest continent. "
                 "Antarctica covers about 14 million km2 and is almost entirely covered by ice. "
                 "It contains 70 percent of the world's fresh water. "
                 "It has no permanent human inhabitants but hosts research stations from many nations. "
                 "The Antarctic Treaty of 1959 bans military activity and reserves it for science. "
                 "Climate change is causing Antarctic ice sheets to melt at an accelerating rate.")
    },
    {
        "id": "geo_great_barrier",
        "title": "Great Barrier Reef",
        "text": ("The Great Barrier Reef is the world's largest coral reef system, "
                 "located off the coast of Queensland, Australia. "
                 "It extends over 2,300 km and contains over 2,900 individual reefs. "
                 "It is home to over 1,500 species of fish, 4,000 types of mollusk, "
                 "and 30 species of whales and dolphins. "
                 "It was declared a UNESCO World Heritage Site in 1981. "
                 "Climate change, ocean acidification, and coral bleaching threaten the reef. "
                 "It supports tourism and fishing industries worth billions of dollars annually.")
    },
    {
        "id": "geo_sahel",
        "title": "Sahel",
        "text": ("The Sahel is a semiarid region stretching across Africa from the Atlantic Ocean "
                 "to the Red Sea. It forms the transition zone between the Sahara to the north "
                 "and the tropical savanna to the south. "
                 "The Sahel spans Senegal, Mali, Niger, Chad, and Sudan. "
                 "It experiences irregular rainfall and frequent droughts. "
                 "The 1970s-1980s Sahel drought caused massive famine and displacement. "
                 "Desertification is advancing southward due to overgrazing and climate change. "
                 "The Great Green Wall initiative aims to restore 100 million hectares of land.")
    },
    {
        "id": "geo_yellowstone",
        "title": "Yellowstone",
        "text": ("Yellowstone National Park is a national park in the United States, "
                 "primarily in Wyoming. "
                 "Established in 1872, it was the world's first national park. "
                 "It sits atop the Yellowstone Caldera, one of the world's largest supervolcanoes. "
                 "Famous features include Old Faithful geyser, hot springs, and fumaroles. "
                 "It is home to bison, elk, wolves, bears, and hundreds of other species. "
                 "The park covers 8,983 km2. "
                 "A Yellowstone eruption would be a catastrophic event affecting North America.")
    },
]


# ============================================================
# INDEX PATH (where to save/load the index)
# ============================================================

INDEX_PATH = os.path.join(project_dir, "search_index.pkl")


# ============================================================
# MAIN INTERACTIVE LOOP
# ============================================================

def startup_and_index():
    """
    Build or load the search index.

    If a saved index exists on disk, load it (fast).
    If not, build it from the article list (slow once, then cached).

    Returns:
        tuple: (indexer, searcher, reranker) ready to use
    """
    print("=" * 53)
    print("  MINI SEARCH ENGINE (50 Wikipedia Articles)")
    print("=" * 53)

    indexer = Indexer()     # Create the indexer
    loaded = indexer.load(INDEX_PATH)   # Try to load saved index

    if not loaded:
        # Build index from scratch
        print("\nFirst run: building index from 50 articles...")
        indexer.load_documents(WIKIPEDIA_ARTICLES)   # Load article list
        indexer.build_index()                         # Encode all articles
        indexer.save(INDEX_PATH)                      # Save for next time
        print("Index saved. Future starts will be faster.")
    else:
        print("Index loaded from disk (faster than rebuilding).")

    # Create searcher and reranker using the built index
    searcher = Searcher(indexer)    # Pass indexer to searcher
    reranker = Reranker()           # Reranker needs no initialization

    print(f"\nReady to search {len(indexer.documents)} articles.")
    print("Topics: Science, History, Technology, Sports, Geography")
    return indexer, searcher, reranker


def format_result(rank, doc_id, score, title, snippet):
    """
    Format a single search result for display.

    Parameters:
        rank    (int):   Result rank (1 = best)
        doc_id  (str):   Document ID
        score   (float): Relevance score from re-ranker
        title   (str):   Article title
        snippet (str):   First ~150 chars of article text

    Returns:
        str: Formatted string for printing
    """
    separator = "-" * 53
    lines = [
        separator,
        f"  [{rank}] {title}  (score: {score:.2f})",
        separator,
        f"  {snippet}",
    ]
    return "\n".join(lines)


def run_search(query, searcher, reranker, indexer):
    """
    Run a complete search for the given query.

    Pipeline:
    1. Searcher retrieves top-20 candidates (bi-encoder style)
    2. Reranker re-scores top-20 and returns top-5 (cross-encoder style)

    Parameters:
        query    (str):     User's search query
        searcher (Searcher): The searcher instance
        reranker (Reranker): The reranker instance
        indexer  (Indexer):  The indexer instance (for looking up text)

    Returns:
        List of (doc_id, score, title, snippet) tuples
    """
    # Step 1: Retrieve top-20 candidates
    candidates = searcher.retrieve(query, k=20)

    if not candidates:
        return []

    # Step 2: Re-rank top-20 to get top-5
    final_results = reranker.rerank(
        query=query,
        candidates=candidates,
        indexer=indexer,
        top_n=5
    )

    return final_results


def main():
    """Main function: startup, index, interactive loop."""

    # Startup: build or load index
    indexer, searcher, reranker = startup_and_index()

    # Interactive search loop
    print("\nType a query to search. Type 'quit' or 'exit' to stop.")
    print("Example queries:")
    print("  gravity black holes")
    print("  ancient egypt pyramids")
    print("  basketball NBA championship")
    print("  amazon river rainforest")
    print("  artificial intelligence machine learning")
    print()

    while True:
        # Show prompt and get user input
        # sys.stdout.flush() ensures prompt appears before input
        print("> ", end="", flush=True)

        try:
            query = input().strip()   # Read user input and remove leading/trailing spaces
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl+D or Ctrl+C gracefully
            print("\nGoodbye!")
            break

        # Check for exit commands
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Skip empty queries
        if not query:
            print("Please type a search query.")
            continue

        # Run the search
        print(f"\nSearching for: '{query}'...")

        results = run_search(query, searcher, reranker, indexer)

        if not results:
            print("No results found. Try different keywords.")
            print()
            continue

        # Display results
        print(f"\nTop {len(results)} results:")
        for rank, (doc_id, score, title, snippet) in enumerate(results, start=1):
            print(format_result(rank, doc_id, score, title, snippet))

        print()   # Blank line before next prompt


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # This block runs when you execute main.py directly.
    # In C#: static void Main(string[] args)
    main()
