import sys


QUERY_INDEX = 0
PRODCUTNAME_INDEX = 1
PRODUCTDESC_INDEX = 2
PRODUCTBRAND_INDEX = 3

separator = '\t'
positives_line = []

end_of_file = False

while not end_of_file:

    positives_line = sys.stdin.readline()
    if not positives_line:
        end_of_file = True

    ptokens = positives_line.rstrip().split(separator)

    inverse_gender_match = False

    if "men" in ptokens[QUERY_INDEX].lower():
        if "women" in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "girl" in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "women" in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True
        if "girl" in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True

    if "boy" in ptokens[QUERY_INDEX].lower():
        if "women" in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "girl" in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "women" in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True
        if "girl" in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True
    if(inverse_gender_match):
        continue


    if "girl"  in ptokens[QUERY_INDEX].lower():
        if "men"  in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "boy"  in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "men"  in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True
        if "boy"  in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True

    if "women"  in ptokens[QUERY_INDEX].lower():
        if "men"  in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "boy"  in ptokens[PRODCUTNAME_INDEX].lower():
            inverse_gender_match = True
        if "men"  in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True
        if "boy"  in ptokens[PRODUCTDESC_INDEX].lower():
            inverse_gender_match = True
    if(inverse_gender_match):
        continue

    print(positives_line.rstrip())
