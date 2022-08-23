# Code based on the implementation https://bitbucket.org/RToroIcarte/lpopl/src/master/ provided by Rodrigo Toro Icarte
"""
This module implements LTL tasks and LTL progression for those tasks

T1 = ("NEXT", "BACON")
T2 = ("UNTIL", "TRUE", "SANDWICH")
T3 = ("UNTIL", "TRUE", ("AND", "EGG", "BACON"))


Definition of prog according to http://www.cs.toronto.edu/~rntoro/docs/LPOPL.pdf:

prog(sigma_i, p) = True  if p in sigma_i
prog(sigma_i, p) = False if p not in sigma_i 
"""

from itertools import chain, combinations


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# Truth symbols
TRUTH_SYMBOLS = {"TRUE", "FALSE"}

# Operators
OPERATORS = {"NOT", "AND", "OR", "UNTIL", "NEXT"}
UNARY_OPERATORS = {"NOT", "NEXT"}
BINARY_OPERATORS = {"AND", "OR", "UNTIL"}


def is_proposition(formula):

    if type(formula) == str and formula not in TRUTH_SYMBOLS.union(OPERATORS):
        return True
    return False


def extract_propositions(formula):
    # Base case
    if type(formula)==str:
        if formula in TRUTH_SYMBOLS:
            return []
        return [formula]
    # Unary operators
    if formula[0] in UNARY_OPERATORS:
        return extract_propositions(formula[1])
    # Binary operators
    return extract_propositions(formula[1]) + extract_propositions(formula[2])

def prog(truth_assignments, formula):
    
    # Base case when TRUE
    if formula == 'TRUE':
        return True
    
    if formula == 'FALSE':
        return False
        
    # Base case when formula is bool
    if type(formula)==bool:
        return formula

    # Base case when formula is proposition
    if is_proposition(formula):
        # Verifying the propostion has a truth assignment
        return formula in truth_assignments

    # Negation
    if formula[0]=="NOT":
        return not prog(truth_assignments, formula[1])

    # And
    if formula[0] == "AND":
        #print(formula)
        P1 = prog(truth_assignments, formula[1])
        P2 = prog(truth_assignments, formula[2])
        #print(P1)
        #print(P2)
        P1_type = type(P1)
        P2_type = type(P2)
        
        if P1_type!=bool and P2_type!=bool:
            return formula

        elif P1_type==bool and P2_type==bool:
            return P1 and P2
        elif P1_type==bool and P1:

            return formula[2]
        elif P2_type==bool and P2:

            return formula[1]
        else:
            return False        

    # Next
    if formula[0] == "NEXT":
        return formula[1]
    
    # Until
    if formula[0] == "UNTIL":

        P1 = prog(truth_assignments, formula[1])
        P2 = prog(truth_assignments, formula[2])
        
        if P2 and type(P2)==bool:
            return True
        if P1:

            if type(P2)==bool:
                return formula
            else:
                return ('UNTIL', formula[1], P2)
        return False

def get_all_progressions(formula):

    P = set(extract_propositions(formula))
    Power_P = [set(x) for x in powerset(P)]

    prog_formulas_list = [formula]
    prog_formulas_set = set(formula)
    
    while len(prog_formulas_list)>0:
        
        next_formula = prog_formulas_list.pop()

        for p in Power_P:

            prog_formula = prog(p, next_formula)
            
            if prog_formula not in prog_formulas_set:
                print(next_formula, p, prog_formula)
                prog_formulas_set.add(prog_formula)
                prog_formulas_list.append(prog_formula)
    
    return prog_formulas_set








def main():
    
    # TEST CASES
    #print(OPERATORS)
    assert(prog({"BACON"}, "BACON"))
    assert(not prog("BACON", "BEANS"))
    assert(prog({"BACON", "EGG"}, ("AND", "EGG", "BACON")))
    assert(not prog({"EGG"}, ("AND", "EGG", "BACON")))
    assert(not prog({"BACON"}, ("AND", "EGG", "BACON")))
    assert(prog({"BACON"}, ("NEXT", "BACON")) == "BACON")
    assert(not prog({"BACON"}, ("NOT", "BACON")))
    
    T3 = ("UNTIL", "TRUE", ("AND", "EGG", "BACON"))
    print(prog({"BACON"}, T3) )
    assert(prog({"BACON"}, T3) == T3)
    assert(prog({"BACON", "EGG"}, T3))

    T4 = ("UNTIL", "BANANA", "BACON")
    
    assert(prog({"BANANA"}, T4) == T4)
    assert(not prog({}, T4))
    assert(prog({"BANANA", "BACON"}, T4))
    T1 = ("NEXT", "BACON")
    print(prog({"BACON"}, T1))
    print('-------------------')
    T = ('UNTIL', 'TRUE', ('AND', 'Fridge', ('UNTIL', 'TRUE', 'Toilet')))

    print(prog({'Fridge'}, T))

    TT = ('UNTIL', 'TRUE', ('UNTIL', 'TRUE', 'Toilet'))
    assert(prog({'Fridge'}, T) == TT)
    print(prog({'Toilet'}, TT))
    assert(prog({'Toilet'}, TT))
    TTT = ('UNTIL', 'TRUE', ('AND', 'Computer',('UNTIL', 'TRUE', ('AND', 'Fridge', ('UNTIL', 'TRUE', 'Toilet')))))
    print(extract_propositions(TTT))

    print([set(x) for x in powerset(set(extract_propositions(TTT)))])
    print(get_all_progressions(TTT))

if __name__ =='__main__':
    main()