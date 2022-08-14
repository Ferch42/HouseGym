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

def prog(truth_assignments, formula):
    
    # Base case when TRUE
    if formula == 'TRUE':
        return True

    # Base case when formula is proposition
    if is_proposition(formula):
        # Verifying the propostion has a truth assignment
        return formula in truth_assignments

    # Negation
    if formula[0]=="NOT":
        return not prog(truth_assignments, formula[1])

    # And
    if formula[0] == "AND":
        return prog(truth_assignments, formula[1]) and prog(truth_assignments, formula[2])

    # Next
    if formula[0] == "NEXT":
        return formula[1]
    
    # Until
    if formula[0] == "UNTIL":
        P1 = prog(truth_assignments, formula[1])
        P2 = prog(truth_assignments, formula[2])

        if P2:
            return True
        if P1:
            return formula
        return False




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

    assert(prog({"BACON"}, T3) == T3)
    assert(prog({"BACON", "EGG"}, T3))

    T4 = ("UNTIL", "BANANA", "BACON")
    
    assert(prog({"BANANA"}, T4) == T4)
    assert(not prog({}, T4))
    assert(prog({"BANANA", "BACON"}, T4))
    T1 = ("NEXT", "BACON")
    print(prog({"BACON"}, T1))



if __name__ =='__main__':
    main()