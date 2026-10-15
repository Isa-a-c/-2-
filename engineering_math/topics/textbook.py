"""Kreyszig 『공업수학 10판 하』 본문 예제의 입력 프리셋입니다.

각 항목은 (절-예제 번호, 인쇄 쪽수, 제목, 주제 ID, 입력값)입니다. 수학 라이브러리를
가져오지 않으며, 검산 값은 tests/test_textbook_examples.py에서 관리합니다.

도약이 있는 함수는 도약점을 적분 격자점에 두고 그 점의 값을 좌우 극한의 평균
(Heaviside(x, 1/2), sign)으로 지정합니다. 사다리꼴 적분의 대칭성으로 교재 값과 일치합니다.
아직 화면에서 지원하지 않는 예제는 UNSUPPORTED에 이유와 함께 기록합니다.
"""
from dataclasses import dataclass, field

PULSE = 'Heaviside(x+1,1/2)-Heaviside(x-1,1/2)'


@dataclass(frozen=True)
class TextbookExample:
    key: str
    page: int
    title: str
    topic: str
    params: dict = field(default_factory=dict)

    @property
    def label(self):
        # '12.6'처럼 예제 번호가 없는 항목은 본문 설명의 대표 문제입니다.
        kind = '예제' if '-' in self.key else '본문'
        return f'교재 {kind} {self.key} · {self.title} ({self.page}쪽)'


EXAMPLES = (
    # ── 11장 Fourier 해석 ─────────────────────────────────────────
    TextbookExample('11.1-1', 6, '주기적인 사각파', 'fourier',
                    {'function': 'k*sign(x)', 'parameters': 'k=1', 'num_points': 5001, 'comparison_N': '1,2,3,10'}),
    TextbookExample('11.1-2', 11, '도약에서의 수렴', 'fourier',
                    {'function': 'k*sign(x)', 'parameters': 'k=1', 'num_points': 5001,
                     'mode': 'Gibbs 확대', 'center': 0., 'width': .5, 'orders': '1,10,50'}),
    TextbookExample('11.2-1', 14, '주기적인 사각파 (p=4)', 'fourier',
                    {'function': f'k*({PULSE})', 'parameters': 'k=1', 'L': '2', 'num_points': 4001}),
    TextbookExample('11.2-2', 14, '사각파의 척도 변환 (p=4)', 'fourier',
                    {'function': 'k*sign(x)', 'parameters': 'k=1', 'L': '2', 'num_points': 4001}),
    TextbookExample('11.2-3', 15, '반파정류기', 'fourier',
                    {'function': 'Heaviside(x,1/2)*E0*sin(w*x)', 'parameters': 'E0=1; w=1', 'L': 'pi',
                     'num_points': 20000, 'comparison_N': '1,2,4,10'}),
    TextbookExample('11.2-5', 17, '톱니파', 'fourier',
                    {'function': 'x+pi', 'num_points': 20000, 'comparison_N': '1,2,3,20'}),
    TextbookExample('11.2-6', 19, '삼각형의 반구간 사인 전개', 'half_range',
                    {'function': 'Piecewise((2*k*x/pi, x<pi/2), (2*k*(pi-x)/pi, True))', 'parameters': 'k=1',
                     'num_points': 20001}),
    TextbookExample('11.2-6', 19, '삼각형의 반구간 코사인 전개', 'half_range',
                    {'function': 'Piecewise((2*k*x/pi, x<pi/2), (2*k*(pi-x)/pi, True))', 'parameters': 'k=1',
                     'num_points': 20001, 'mode': '코사인 · 짝수 확장'}),
    TextbookExample('11.3-1', 23, '비사인 주기 외력의 강제진동', 'fourier_forced',
                    {'function': 'Piecewise((x+pi/2, x<0), (-x+pi/2, True))', 'N': 9, 'num_points': 20001,
                     'mass': 1., 'damping': .05, 'stiffness': 25., 'end_time': 20.}),
    TextbookExample('11.4-1', 28, '톱니파의 최소제곱오차 (N=20)', 'trig_approximation',
                    {'function': 'x+pi', 'N': 20, 'num_points': 20001, 'perturbation': 0.}),
    TextbookExample('11.5-1', 30, '진동하는 현의 고유함수', 'sturm_liouville', {'function': 'x', 'N': 6}),
    TextbookExample('11.5-2', 31, '직교함수 sin mx와 노름 √π (구간 길이 2π)', 'sturm_liouville',
                    {'mode': 'Periodic', 'L': '2*pi', 'function': 'x', 'N': 7, 'num_points': 20001}),
    TextbookExample('11.5-4', 34, 'Legendre 다항식의 직교성', 'orthogonal_series',
                    {'function': 'x^2', 'N': 6, 'num_points': 20001}),
    TextbookExample('11.6-1', 36, 'sin πx의 Fourier–Legendre 급수', 'orthogonal_series',
                    {'function': 'sin(pi*x)', 'N': 12, 'num_points': 20001}),
    TextbookExample('11.6-3', 38, '1−x²의 Fourier–Bessel 급수', 'orthogonal_series',
                    {'function': '1-x^2', 'mode': 'Bessel', 'L': '1', 'N': 6, 'num_points': 20001}),
    TextbookExample('11.7-1', 42, '사각파의 주기 2L→∞ (그림 280)', 'fourier_integral',
                    {'function': PULSE, 'mode': '주기 2L→∞ 스펙트럼', 'periods': '4, 8, 16', 'extent': 8.,
                     'cutoff': 12., 'num_points': 8001}),
    TextbookExample('11.7-2', 45, '단일 펄스와 Gibbs 현상 (a=16)', 'fourier_integral',
                    {'function': PULSE, 'extent': 8., 'cutoff': 16., 'num_points': 8001, 'frequency_points': 1601}),
    # 11.7 예제 3은 Fourier 코사인·사인 적분의 A(w), B(w)입니다. x>0의 e^(−kx)를 우·기확장하여 적분합니다.
    TextbookExample('11.7-3', 47, 'Laplace 적분: 코사인 적분 A(w)', 'fourier_integral',
                    {'function': 'exp(-k*abs(x))', 'parameters': 'k=1', 'extent': 40., 'num_points': 8193}),
    TextbookExample('11.7-3', 48, 'Laplace 적분: 사인 적분 B(w)', 'fourier_integral',
                    {'function': 'sign(x)*exp(-k*abs(x))', 'parameters': 'k=1', 'extent': 40., 'num_points': 8193}),
    TextbookExample('11.8-1', 51, '사각 펄스의 코사인 변환', 'sine_cosine_transform',
                    {'function': 'k*Heaviside(a-x,1/2)', 'parameters': 'k=1; a=1'}),
    TextbookExample('11.8-1', 51, '사각 펄스의 사인 변환', 'sine_cosine_transform',
                    {'function': 'k*Heaviside(a-x,1/2)', 'parameters': 'k=1; a=1', 'mode': '사인'}),
    TextbookExample('11.8-2', 51, '지수함수의 코사인 변환', 'sine_cosine_transform',
                    {'function': 'exp(-x)', 'extent': 40., 'num_points': 8193}),
    TextbookExample('11.8-3', 53, '연산공식의 응용 e^(−ax)', 'sine_cosine_transform',
                    {'function': 'exp(-a*x)', 'parameters': 'a=2', 'extent': 40., 'num_points': 8193}),
    TextbookExample('11.9-1', 56, '사각 펄스의 Fourier 변환', 'continuous_fourier',
                    {'function': PULSE, 'extent': 8., 'num_points': 8001}),
    TextbookExample('11.9-2', 56, '단측 지수함수의 Fourier 변환', 'continuous_fourier',
                    {'function': 'Heaviside(x,1/2)*exp(-a*x)', 'parameters': 'a=1', 'extent': 40., 'num_points': 8001}),
    TextbookExample('11.9-3', 59, 'x e^(−x²)의 Fourier 변환', 'continuous_fourier',
                    {'function': 'x*exp(-x^2)'}),
    # 예제 4(DFT·Fourier 행렬)와 예제 5(FFT 짝·홀 분해)는 같은 데이터이며 한 화면에 함께 나옵니다.
    TextbookExample('11.9-4', 62, 'DFT와 FFT의 짝·홀 분해 (예제 4·5, N=4)', 'dft', {'data': '0, 1, 4, 9'}),
    TextbookExample('11.9', 60, '합성곱 정리 (정리 4)', 'continuous_fourier',
                    {'mode': '합성곱 정리', 'function': 'exp(-x^2/2)', 'second': 'exp(-x^2/2)', 'extent': 12.,
                     'num_points': 4001}),
    # ── 12장 편미분방정식 ─────────────────────────────────────────
    TextbookExample('12.1-1', 74, 'Laplace 방정식의 해 x²−y²', 'pde_basics', {'candidate': 'x^2-y^2'}),
    TextbookExample('12.1-1', 74, 'Laplace 방정식의 해 eˣcos y', 'pde_basics', {'candidate': 'exp(x)*cos(y)'}),
    TextbookExample('12.1-1', 74, 'Laplace 방정식의 해 sin x cosh y', 'pde_basics', {'candidate': 'sin(x)*cosh(y)'}),
    TextbookExample('12.1-2', 75, 'u_xx − u = 0을 상미분방정식처럼 풀기', 'pde_basics',
                    {'mode': '상미분방정식처럼 풀기', 'variable': 'x', 'a': '1', 'b': '0', 'c': '-1'}),
    TextbookExample('12.1-3', 75, 'u_xy = −u_x를 p=u_x로 치환하여 풀기', 'pde_basics',
                    {'mode': '상미분방정식처럼 풀기', 'variable': 'y', 'substitution': 'p = u_x로 치환',
                     'a': '0', 'b': '1', 'c': '1'}),
    TextbookExample('12.3-1', 84, '초기변위가 삼각형인 진동현', 'wave',
                    {'initial': 'Piecewise((2*k*x/l, x<l/2), (2*k*(l-x)/l, True))', 'parameters': 'k=0.1',
                     'N': 50, 'num_points': 2001, 'end_time': 2., 'time_points': 401}),
    TextbookExample('12.4-1', 90, '특성으로 얻은 D’Alembert 해', 'pde_classification',
                    {'A': '1', 'B': '0', 'C': '-1'}),
    TextbookExample('12.6-1', 95, '구리 막대의 사인형 초기온도', 'heat',
                    {'initial': '100*sin(pi*x/l)', 'length': 80., 'alpha': 1.158, 'N': 5,
                     'end_time': 800., 'time_points': 501}),
    TextbookExample('12.6-2', 96, '감쇠속도 (n=3 모드)', 'heat',
                    {'initial': '100*sin(3*pi*x/l)', 'length': 80., 'alpha': 1.158, 'N': 5,
                     'end_time': 100., 'time_points': 501}),
    TextbookExample('12.6-3', 97, '막대의 삼각형 초기온도', 'heat',
                    {'initial': 'Piecewise((x, x<l/2), (l-x, True))', 'length': 3.141592653589793,
                     'alpha': 1., 'N': 50, 'num_points': 2001, 'end_time': 2.}),
    TextbookExample('12.6-4', 97, '양 끝이 단열된 막대 (f=x, 고유값 0)', 'heat',
                    {'mode': '양 끝 단열', 'initial': 'x', 'length': 3.141592653589793, 'alpha': 1.,
                     'N': 50, 'num_points': 2001, 'end_time': 3.}),
    TextbookExample('12.6-5', 98, '양 끝이 단열된 삼각형 초기온도', 'heat',
                    {'mode': '양 끝 단열', 'initial': 'Piecewise((x, x<l/2), (l-x, True))',
                     'length': 3.141592653589793, 'alpha': 1., 'N': 50, 'num_points': 2001, 'end_time': 3.}),
    TextbookExample('12.6', 99, '직사각형의 Dirichlet 문제 (그림 296)', 'rectangle_laplace',
                    {'top': '25', 'length': 24., 'height': 24., 'N': 80, 'num_points': 121}),
    TextbookExample('12.7-1', 104, '무한히 긴 막대의 온도', 'infinite_heat',
                    {'initial': f'U0*({PULSE})', 'parameters': 'U0=100', 'alpha': 1., 'extent': 3.,
                     'num_points': 241, 'end_time': 8., 'time_points': 65}),
    TextbookExample('12.7-4', 107, '반무한 막대의 Fourier 사인 변환 해 (그림 300)', 'infinite_heat',
                    {'mode': '반무한 막대 · u(0,t)=0', 'initial': 'Heaviside(1-x,1/2)', 'alpha': 1., 'extent': 2.,
                     'num_points': 201, 'end_time': 1.5, 'time_points': 151}),
    TextbookExample('12.9-1', 115, '정사각형 막의 중복 고유값 (u₁₂+u₂₁)', 'rectangle_membrane',
                    {'initial': 'sin(pi*x)*sin(2*pi*y)+B*sin(2*pi*x)*sin(pi*y)', 'parameters': 'B=1',
                     'N': 3, 'end_time': 1.}),
    TextbookExample('12.9-2', 117, '직사각형 막의 진동', 'rectangle_membrane',
                    {'initial': '0.1*(4*x-x^2)*(2*y-y^2)', 'length': 4., 'height': 2.,
                     'speed': 2.23606797749979, 'N': 7, 'num_points': 161, 'end_time': 2.}),
    TextbookExample('12.10-1', 124, '원형 막의 진동', 'disk_membrane',
                    {'initial': '1-r^2', 'radius': 1., 'speed': 2., 'radial_count': 10, 'angular_count': 0,
                     'end_time': 2., 'time_points': 101}),
    TextbookExample('12.11-1', 131, '구형 축전기의 내부 퍼텐셜', 'coordinate_potential',
                    {'mode': '구 · 축대칭', 'boundary': '110*Heaviside(mu,1/2)', 'N': 15}),
    TextbookExample('12.11-1', 132, '구형 축전기의 외부 퍼텐셜', 'coordinate_potential',
                    {'mode': '구 · 축대칭 외부', 'boundary': '110*Heaviside(mu,1/2)', 'N': 15}),
    TextbookExample('12.11-2', 133, '더 단순한 경우 cos 2φ', 'coordinate_potential',
                    {'mode': '구 · 축대칭', 'boundary': '2*mu^2-1', 'N': 4}),
    TextbookExample('12.12-1', 135, '반무한 현', 'pde_laplace',
                    {'boundary': 'Piecewise((sin(t), t<2*pi), (0, True))', 'speed': 1., 'extent': 30.,
                     'end_time': 20., 'time_points': 201, 'num_points': 1501}),
)

# 본문 예제 중 아직 화면 기능이 없는 항목입니다. 기능을 추가하면 EXAMPLES로 옮깁니다.
UNSUPPORTED = {
}


def examples_for(topic_id):
    return {example.label: dict(example.params) for example in EXAMPLES if example.topic == topic_id}
