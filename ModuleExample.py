"""
ModuleExample                   -  맨 앞에 해당 모듈에 대한 정보를 적어주세요

사칙 연산을 수행해주는 모듈

Functions                       -  해당 모듈에서 export할 함수들을 적어주세요 (파이썬은 매개변수와 리턴값의 자료형을 명시하지 않지만, 오류 최소화를 위해 적어주세요)

int add(int, int)       // 덧셈
int sub(int, int)       // 뺄셈
int mul(int, int)       // 곱셈
double div(int, int)    // 나눗셈

"""

def add(n, m):
    return n + m

def sub(n, m):
    return n - m

def mul(n, m):
    return n * m

def div(n, m):
    if m == 0:
        print("error")
    return n / m

"""
이렇게 모듈을 작성하고 나면 해당 모듈을 사용할 다른 코드에서

import ModuleExample 입력 후

a = ModuleExample.add(b, c) 와 같이 사용 가능합니다

"""
