def demo_0():
    locations = ["Midwestern", "Eastern", "Southern", "Western"]
    ages = ["twenties", "thirties", "forties", "fifties", "sixties"]
    gender = ["man", "woman"]
    urbanicities = ["a rural", "an exurban" , "a suburban", "an urban"]
    educations = ["Some High School","High School", "an Associate’s Degree", "Some College", "College", "a Postgraduate Degree"]

    def sentence_constructor(gender, age, urbanicity, location, education, flexible_attribute):
        cur_str =   (
            f"You are a {gender} in their {age} from {urbanicity} part of the {location} United States.\n"
            f"Your highest level of educational attainment is {education}.\n"
        )

        if flexible_attribute is not None:
            cur_str += flexible_attribute + '\n'
        
        return cur_str
    
    return sentence_constructor, [gender, ages, urbanicities, locations, educations]

def demo_1():
    identity = ["United States"]

    def sentence_constructor(ident, flexible_attribute):
        return f"You are from the {ident}"

    return sentence_constructor, [identity]


def demo_shim(demo_idx):
    if demo_idx == 0:
        return demo_0()
    elif demo_idx == 1:
        return demo_1()
