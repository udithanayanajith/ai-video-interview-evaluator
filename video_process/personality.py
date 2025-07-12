import random

# Simulate Big Five trait predictions per video
def simulate_big_five_scores(video_path):
    print(f"Simulating Big Five traits for video: {video_path}")
    import hashlib
    seed = int(hashlib.md5(video_path.encode()).hexdigest(), 16) % 100000
    random.seed(seed)  # Consistent per video
    return {
        "Openness": round(random.uniform(0.5, 1.0), 2),
        "Conscientiousness": round(random.uniform(0.5, 1.0), 2),
        "Extraversion": round(random.uniform(0.3, 0.9), 2),
        "Agreeableness": round(random.uniform(0.4, 1.0), 2),
        "Neuroticism": round(random.uniform(0.1, 0.5), 2),
    }


# Score roles based on Big Five traits
def score_roles(traits):
    scores = {
        'software_engineer': traits["Conscientiousness"] * 0.4 +
                             traits["Openness"] * 0.3 +
                             (1 - traits["Neuroticism"]) * 0.3,

        'associate_software_engineer': traits["Conscientiousness"] * 0.3 +
                                       traits["Agreeableness"] * 0.2 +
                                       traits["Openness"] * 0.2 +
                                       (1 - traits["Neuroticism"]) * 0.3,

        'it_intern': traits["Openness"] * 0.4 +
                     traits["Extraversion"] * 0.2 +
                     traits["Agreeableness"] * 0.1 +
                     (1 - traits["Neuroticism"]) * 0.3,
    }

    return {role: int(round(score * 100)) for role, score in scores.items()}

# Average traits over multiple videos
def average_traits(traits_list):
    avg = {}
    total = len(traits_list)
    for trait in traits_list[0].keys():
        avg[trait] = round(sum(t[trait] for t in traits_list) / total, 2)
    return avg
