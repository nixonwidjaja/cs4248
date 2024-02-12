'''
    NUS CS4248 Assignment 1 - Objective 1 (Regexes)
    Enter your expressions into the sections below 
    and run this file for a preliminary test.

    Example usage: 
    >> python3 obj1_regex.py

    If you spot any bug during testing, send feedback to jiatong.han@u.nus.edu
'''
import re

# TODO: Place your answers between the quotes below.
R1 = r"xxx"
R2 = r"xxx"
R3 = r"xxx"
R4 = r"xxx"
R5 = r"xxx"  # bonus
run_bonus = False  # set to True if you want to test bonus question

# Minitests: {match_string: expected_outcome}
# For your own testing: append with more test strings
minitest1 = {'cbc': True, '01230': True, 'a': False, 'abc': False}
minitest2 = {'otto': False, 'dttjttd': True, '-++-': False, 'abcd': True}
minitest3 = {'abba': False, 'dttjttd': True}
minitest4 = {'+..+': True, 'trillion': False}
minitest5 = {':)': True, ':-(': True, ';p': True, '(:': True}

class RegexTest:
    def result(self, matched: bool):
        if matched:
            return 'matched = True'
        else:
            return 'matched = False'

    def minitest_r1(self):
        for (string, matched) in minitest1.items():
            assert (re.search(R1, string) is not None) == matched, \
                f'''minitest r1 failed with re.search(r"{R1}", "{string}"). 
                Expected result: {self.result(matched)}, actual result: {self.result(not matched)}'''
        print("test r1 passed")

    def minitest_r2(self):
        for (string, matched) in minitest2.items():
            assert (re.search(R2, string) is not None) == matched, \
               f'''minitest r2 failed with re.search(r"{R2}", "{string}"). 
               Expected result: {self.result(matched)}, actual result: {self.result(not matched)}'''
        print("test r2 passed")

    def minitest_r3(self):
        for (string, matched) in minitest3.items():
            assert (re.search(R3, string) is not None) == matched, \
               f'''minitest r3 failed with re.search(r"{R3}", "{string}"). 
               Expected result: {self.result(matched)}, actual result: {self.result(not matched)}'''
        print("test r3 passed")

    def minitest_r4(self):
        for (string, matched) in minitest4.items():
            assert (re.search(R4, string) is not None) == matched, \
               f'''minitest r4 failed with re.search(r"{R4}", "{string}"). 
               Expected result: {self.result(matched)}, actual result: {self.result(not matched)}'''
        print("test r4 passed")

    def minitest_r5(self):
        for (string, matched) in minitest5.items():
            assert (re.search(R5, string) is not None) == matched, \
            f'''minitest r5 failed with re.search(r"{R5}", "{string}"). 
               Expected result: {self.result(matched)}, actual result: {self.result(not matched)}'''
        print("test r5 passed")

    def run_all_tests(self):
        # comment out unwanted tests
        self.minitest_r1() 
        self.minitest_r2()
        self.minitest_r3()
        self.minitest_r4()
        if run_bonus:
            self.minitest_r5()
        else:
            print("bonus: not run")

if __name__ == "__main__":
    RegexTest().run_all_tests()
