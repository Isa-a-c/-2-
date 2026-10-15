"""변수·인자 수·복소수 정책을 주제별로 지정하는 제한된 SymPy 입력."""
import ast
from functools import lru_cache
import re
import numpy as np
import sympy as sp

FUNCTIONS = {
    'sin': (sp.sin,1,1), 'cos': (sp.cos,1,1), 'tan': (sp.tan,1,1),
    'exp': (sp.exp,1,1), 'log': (sp.log,1,2), 'sqrt': (sp.sqrt,1,1),
    'abs': (sp.Abs,1,1), 'Abs': (sp.Abs,1,1), 'sign': (sp.sign,1,1),
    'sinh': (sp.sinh,1,1), 'cosh': (sp.cosh,1,1), 'tanh': (sp.tanh,1,1),
    'Heaviside': (sp.Heaviside,1,2), 'Piecewise': (sp.Piecewise,1,16),
    'besselj': (sp.besselj,2,2), 'bessely': (sp.bessely,2,2),
    'gamma': (sp.gamma,1,1), 'erf': (sp.erf,1,1), 'atan2': (sp.atan2,2,2),
    'asin': (sp.asin,1,1), 'acos': (sp.acos,1,1), 'atan': (sp.atan,1,1),
    'arcsin': (sp.asin,1,1), 'arccos': (sp.acos,1,1), 'arctan': (sp.atan,1,1),
    'asinh': (sp.asinh,1,1), 'acosh': (sp.acosh,1,1), 'atanh': (sp.atanh,1,1),
    'ln': (sp.log,1,2), 'sinc': (sp.sinc,1,1),
    'floor': (sp.floor,1,1), 'ceil': (sp.ceiling,1,1), 'ceiling': (sp.ceiling,1,1),
    'erfc': (sp.erfc,1,1), 'erfi': (sp.erfi,1,1),
    'besseli': (sp.besseli,2,2), 'besselk': (sp.besselk,2,2),
    'beta': (sp.beta,2,2), 'loggamma': (sp.loggamma,1,1),
    'DiracDelta': (sp.DiracDelta,1,2),
    'And': (sp.And,2,16), 'Or': (sp.Or,2,16), 'Not': (sp.Not,1,1),
    'Eq': (sp.Eq,2,2), 'Ne': (sp.Ne,2,2), 'Lt': (sp.Lt,2,2),
    'Le': (sp.Le,2,2), 'Gt': (sp.Gt,2,2), 'Ge': (sp.Ge,2,2),
}
NODES = (ast.Expression,ast.BinOp,ast.UnaryOp,ast.Call,ast.Name,ast.Load,
         ast.Constant,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.UAdd,ast.USub,
         ast.Tuple,ast.Compare,ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq,
         ast.BitAnd,ast.BitOr,ast.Invert)


class SymbolicComparisons(ast.NodeTransformer):
    """Python의 == 평가 대신 SymPy의 기호 관계식을 생성합니다."""
    names = {ast.Lt: '__lt', ast.LtE: '__le', ast.Gt: '__gt', ast.GtE: '__ge',
             ast.Eq: '__eq', ast.NotEq: '__ne'}

    def visit_Compare(self, node):
        self.generic_visit(node)
        terms = [node.left, *node.comparators]
        comparisons = [ast.Call(func=ast.Name(id=self.names[type(op)], ctx=ast.Load()),
                               args=[terms[index], terms[index+1]], keywords=[])
                       for index, op in enumerate(node.ops)]
        if len(comparisons) == 1:
            return comparisons[0]
        return ast.Call(func=ast.Name(id='__and', ctx=ast.Load()), args=comparisons, keywords=[])


def symbols_for(variables, allow_complex=False):
    return tuple(sp.Symbol(name) if allow_complex else sp.Symbol(name,real=True) for name in variables)


def parse_expression(text, variables=('x',), allow_complex=False):
    # 수학 입력의 ^는 거듭제곱으로 해석합니다. 곱셈은 *를 명시합니다.
    if isinstance(text, str):
        text = text.strip().replace('^', '**')
    return _parse(text,tuple(variables),bool(allow_complex))


@lru_cache(maxsize=64)
def _parse(text, variables, allow_complex):
    if not isinstance(text,str) or not 1 <= len(text.strip()) <= 1000:
        raise ValueError('수식은 1~1000자 문자열로 입력하십시오.')
    if not variables or len(set(variables)) != len(variables) or any(
        not re.fullmatch('[A-Za-z][A-Za-z0-9]*',v) or v in FUNCTIONS or v in ['pi','E','I','True','False'] for v in variables):
        raise ValueError('변수 이름이 유효하지 않습니다.')
    if not re.fullmatch(r'[A-Za-z0-9_+\-*/().,<>=!&|~\s]+',text):
        raise ValueError('지원하지 않는 문자가 있습니다.')
    local={name:item[0] for name,item in FUNCTIONS.items()}
    local.update(zip(variables,symbols_for(variables,allow_complex)))
    if 'l' in variables and 'ell' not in variables:
        # 길이 변수는 v1.9.2부터 l입니다. 예전 설정 파일의 ell도 같은 기호로 읽습니다.
        local['ell'] = local['l']
    local.update(pi=sp.pi,E=sp.E)
    if allow_complex: local['I']=sp.I
    tree=ast.parse(text,mode='eval')
    if len(list(ast.walk(tree)))>250: raise ValueError('수식이 너무 복잡합니다.')
    for node in ast.walk(tree):
        if not isinstance(node,NODES): raise ValueError('지원하지 않는 수식 문법입니다.')
        if isinstance(node,ast.Name) and node.id not in local:
            raise ValueError(f'지원하지 않는 이름: {node.id}')
        if isinstance(node,ast.Constant) and (type(node.value) not in (int,float,bool) or abs(node.value)>1e100):
            raise ValueError('상수가 유효하지 않습니다.')
        if isinstance(node,ast.Call):
            if not isinstance(node.func,ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ValueError('지원 함수만 호출할 수 있습니다.')
            _,minimum,maximum=FUNCTIONS[node.func.id]
            if not minimum<=len(node.args)<=maximum: raise ValueError('함수의 인자 개수를 확인하십시오.')
            if node.func.id=='Piecewise' and any(not isinstance(arg,ast.Tuple) or len(arg.elts)!=2 for arg in node.args):
                raise ValueError('Piecewise에는 (식, 조건) 쌍을 입력하십시오.')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            if isinstance(node.right,ast.Constant) and abs(node.right.value)>1000:
                raise ValueError('직접 입력하는 지수의 절댓값은 1000 이하여야 합니다.')
    local.update(__lt=sp.Lt, __le=sp.Le, __gt=sp.Gt, __ge=sp.Ge,
                 __eq=sp.Eq, __ne=sp.Ne, __and=sp.And)
    symbolic_tree = ast.fix_missing_locations(SymbolicComparisons().visit(tree))
    expression=sp.sympify(ast.unparse(symbolic_tree),locals=local)
    if not isinstance(expression,sp.Expr) or expression.free_symbols-set(symbols_for(variables,allow_complex)):
        raise ValueError('지정된 변수의 수식을 입력하십시오.')
    if expression.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ValueError('수식에 무한대 또는 정의되지 않은 값이 있습니다.')
    if not allow_complex and expression.has(sp.I):
        raise ValueError('이 주제는 실수 함수만 지원합니다.')
    return expression


@lru_cache(maxsize=32)
def compiled(expression, variables=('x',), allow_complex=False):
    # 기본 함수는 NumPy만 사용하며 특수함수에 필요한 경우에만 SciPy를 불러온다.
    special = (sp.besselj, sp.bessely, sp.besseli, sp.besselk, sp.gamma,
               sp.loggamma, sp.beta, sp.erf, sp.erfc, sp.erfi)
    modules=['scipy','numpy'] if expression.has(*special) else 'numpy'
    return sp.lambdify(symbols_for(variables,allow_complex),expression,modules=modules)


def evaluate(expression, values, variables=('x',), allow_complex=False):
    if expression.has(sp.DiracDelta):
        raise ValueError('DiracDelta는 분포이므로 일반 함수의 수치 표본으로 평가할 수 없습니다. 라플라스 기호 계산을 사용하십시오.')
    arrays=np.broadcast_arrays(*[np.asarray(value, dtype=complex if allow_complex else None) for value in values])
    with np.errstate(all='ignore'):
        result=np.asarray(compiled(expression,tuple(variables),allow_complex)(*arrays))
    if not allow_complex and np.iscomplexobj(result):
        raise ValueError('복소수 값이 발생했습니다. 실수 함수를 입력하십시오.')
    result=np.array(np.broadcast_to(result,arrays[0].shape),dtype=complex if allow_complex else float,copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError('표본에 NaN 또는 inf가 있습니다. 함수와 구간을 확인하십시오.')
    return result


def real_constant(text):
    expression=parse_expression(str(text))
    if expression.free_symbols: raise ValueError('변수가 없는 실수 상수를 입력하십시오.')
    value=float(expression)
    if not np.isfinite(value): raise ValueError('유한한 상수를 입력하십시오.')
    return value


def parse_parameters(text):
    """A=2; w=3*pi처럼 독립적인 유한 실수 상수를 선언합니다."""
    if not isinstance(text, str) or len(text) > 1000:
        raise ValueError('매개변수는 1000자 이하의 문자열로 입력하십시오.')
    if not text.strip():
        return {}
    items = text.split(';')
    if len(items) > 16:
        raise ValueError('매개변수는 16개 이하로 입력하십시오.')
    reserved = set(FUNCTIONS) | {'x', 't', 's', 'l', 'ell', 'pi', 'E', 'I', 'True', 'False'}
    parameters = {}
    for item in items:
        if item.count('=') != 1:
            raise ValueError('매개변수 형식은 A=2; w=3*pi입니다. 세미콜론으로 구분하십시오.')
        name, value = (part.strip() for part in item.split('='))
        if not re.fullmatch('[A-Za-z][A-Za-z0-9]*', name) or name in reserved or name in parameters:
            raise ValueError(f'매개변수 이름이 중복되었거나 사용할 수 없습니다: {name}')
        constant = parse_expression(value)
        if constant.free_symbols or constant.is_real is not True or constant.is_finite is not True:
            raise ValueError(f'{name}: 다른 매개변수에 의존하지 않는 유한 실수 상수를 입력하십시오.')
        if not np.isfinite(float(constant)) or abs(float(constant)) > 1e100:
            raise ValueError(f'{name}: 매개변수 크기가 너무 큽니다.')
        parameters[name] = constant
    return parameters


def parameter_expression(text, parameter_text='', variables=('x',)):
    """허용된 매개변수를 기호로 해석한 뒤 정확한 상수로 치환합니다."""
    parameters = parse_parameters(parameter_text)
    expression = parse_expression(text, (*variables, *parameters))
    substitutions = dict(zip(symbols_for(tuple(parameters)), parameters.values()))
    resolved = expression.subs(substitutions)
    if resolved.has(sp.zoo, sp.oo, -sp.oo, sp.nan):
        raise ValueError('매개변수 대입 후 정의되지 않은 값이 발생했습니다.')
    if resolved.has(sp.I):
        raise ValueError('매개변수 대입 후 복소수 값이 발생했습니다. 실수 수식을 입력하십시오.')
    return resolved
