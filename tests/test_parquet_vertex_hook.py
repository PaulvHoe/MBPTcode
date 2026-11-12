"""The reducible-vertex hook and the vertex damping of both parquet drivers.

`vertex_builder` (name, spectra, eta_2b) -> Phi replaces Quack's static
reducible vertex (`restricted.static_reducible_vertex`,
`spin_orbital.static_reducible_vertex_so`); `damping` d is the weight of the
previous vertices, Phi <- (1 - d) Phi_new + d Phi_old, evaluated as Quack
evaluates its mixing with alpha = 1 - d, fl(fl(alpha Phi_new) +
fl((1 - alpha) Phi_old)). Held on water/6-31G:

* a builder that returns Quack's static vertex (a function of its own, a
  functools.partial of the default) gives the default run, every scheme,
  with and without two-body DIIS: the same eqp, Z and vertices (two runs, so
  to a few ulp as `tests.pins.assert_same` holds them), counts and flags; the
  options record names the builder;
* the arrays a builder hands out (cached between calls) are only read: after
  the run they hold the bits they were returned with, at the default damping
  and at 0.4 (where the spin-orbital default, alpha = 1, would hide a write);
* a NaN no kernel reads (closed shell pp_triplet[0, 0, 0, 0], an occupied
  pair i = j; spin orbitals eh[0, 0, 0, 0], a pair p = q) is kept and
  flagged, the change of that vertex passes over it as gfortran's maxval
  does, and the run without DIIS is the default one (with DIIS the NaN
  enters the Gram matrix);
* the history's two-body rcond is that of Quack's DIIS fed Quack's errors,
  damped - old (a uniform scale of the errors, new - old, leaves the
  energies and only moves the rcond);
* the damping: the default is the explicit value (0.75 for the closed-shell
  driver and both qsPA, Quack's alpha 0.25; 0 for the spin-orbital FLEX,
  osPA and evPA, G_evParquet's alpha 1), the same run; FLEX's vertices are
  alpha times the new ones to the bit, and the second iteration's are
  Quack's mixing of the builder's two vertices to the bit (elementwise
  products and sums, which round alike on any machine); damping 0.4
  against Quack's R_evParquet and R_qsParquet compiled with alpha = 0.6
  (QUACK_DAMPED, recorded from the Fortran: osPA with and without two-body
  DIIS, evPA by Newton and linearized, qsPA).

Run: python tests/test_parquet_vertex_hook.py, or under pytest.
"""
import functools
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from src.Base.numerics.constants import PARQUET_SO_EV_VERTEX_DAMPING, PARQUET_VERTEX_DAMPING
from src.SingleReference.Parquet import restricted, spin_orbital
from src.SingleReference.Parquet.diis import QuackDIIS
from src.SingleReference.Parquet.restricted import static_reducible_vertex
from src.SingleReference.Parquet.spin_orbital import static_reducible_vertex_so
from tests.test_parquet_restricted_vs_quack import (NEWTON, assert_matches_quack, bars,
                                                    mo_integrals, qs_inputs, rhf, run_case)
from tests.pins import assert_same
from tests.test_parquet_spin_adaptation import both_layouts

#: The damping of the Fortran runs compiled with alpha = 1 - DAMPING.
DAMPING = 0.4
#: Relative bar on a recorded DIIS rcond against QuackDIIS on the same pairs:
#: the loop and the reference sum the Gram matrix in other orders (an rcond
#: from errors scaled by 1/alpha differs by 7.3 times).
RCOND_RTOL = 1e-10

T = True
#: Runs of every scheme, short: (scheme, options); the quasiparticle equation by
#: Quack's Newton (`NEWTON`), cheap beside the root finder, as the vertex hook
#: does not reach it.
SCHEMES = {
    'flex': ('flex', dict(eta_2b=1.0, **NEWTON)),
    'os': ('os', dict(eta_2b=1.0, max_it_2b=6, conv_2b=1e-12, **NEWTON)),
    'ev': ('ev', dict(eta_2b=1.0, max_it_2b=4, conv_2b=1e-12, max_it_1b=2, **NEWTON)),
    'qs': ('qs', dict(eta_2b=1.0, max_it_2b=4, conv_2b=1e-12, max_it_1b=2)),
}
#: The damped cases, as `test_parquet_restricted_vs_quack.CASES` runs them.
DAMPED = {}
for _mol in ('water', 'ammonia'):
    DAMPED.update({
        f'{_mol}/os_tda_s1_dmp04': ('ev', 'os', dict(tda_eh=T, tda_pp=T, eta_1b=50.0, eta_2b=1.0,
                                                     conv_2b=1e-7, max_it_2b=400,
                                                     damping=DAMPING, **NEWTON), 'exact'),
        f'{_mol}/os_tda_s1_d5_dmp04': ('ev', 'os', dict(tda_eh=T, tda_pp=T, eta_1b=50.0,
                                                        eta_2b=1.0, conv_2b=1e-7, max_it_2b=400,
                                                        max_diis_2b=5, damping=DAMPING,
                                                        **NEWTON), 'diis'),
        f'{_mol}/ev_n50_dmp04': ('ev', 'ev', dict(tda_eh=T, tda_pp=T, eta_1b=50.0, eta_2b=1.0,
                                                  conv_2b=1e-8, max_it_2b=400, conv_1b=1e-7,
                                                  max_it_1b=6, damping=DAMPING, **NEWTON),
                                 'exact'),
        f'{_mol}/ev_l50_dmp04': ('ev', 'ev', dict(tda_eh=T, tda_pp=T, eta_1b=50.0, eta_2b=1.0,
                                                  conv_2b=1e-8, max_it_2b=400, conv_1b=1e-7,
                                                  max_it_1b=6, qp_solver='linearized',
                                                  damping=DAMPING), 'exact'),
        f'{_mol}/qs_tda_s1_dmp04': ('qs_dsyev', 'qs', dict(tda_eh=T, tda_pp=T, eta_1b=1.0,
                                                           eta_2b=1.0, reg_pa=T, conv_1b=1e-6,
                                                           max_it_1b=4, conv_2b=1e-7,
                                                           max_it_2b=400, max_diis_1b=3,
                                                           damping=DAMPING), 'exact'),
    })

# Quack's R_evParquet and R_qsParquet (the latter with the omega = 0d0 its four
# R_*_Phi calls miss) of the paper-era tree (git e3528a40), compiled with
# alpha = 0.6d0 in copies that print their numbers in full precision and are
# otherwise unchanged, one thread, on the inputs of
# test_parquet_restricted_vs_quack (qsPA with Quack's own dsyev
# orthogonalizer, 'qs_dsyev'), Ha.
QUACK_DAMPED = {
    'water/os_tda_s1_dmp04': {
        'n_it_1b': 1,
        'n_it_2b': [68],
        'err_1b': [0.5009176851435377],
        'eqp': [-20.059516031111766, -1.2988117702301465, -0.6747524452960445, -0.5026815112569699,
                -0.43313281518457714, 0.18850217942226222, 0.2821298764892595, 1.0270078622032182,
                1.1189335073159201, 1.1164521013094098, 1.1671072598794836, 1.2810946994110288,
                1.6768587651946039],
        'sigc': [0.6984642916155489, 0.055913274244500925, 0.037234227029072635,
                 0.06209012421770395, 0.07381963285382363, -0.015482334095165163,
                 -0.018136314163269565, -0.031742517541656703, -0.049138805263598304,
                 -0.09011187428037283, -0.05298704920145297, -0.14462329564186846,
                 -0.017399025011791042],
        'z': [0.7983114095382614, 0.7682316311079699, 0.950852320542427, 0.937198325801715,
              0.9303706840288185, 0.9834038958788976, 0.9759378567920338, 0.9545989770034041,
              0.9328668353995759, 0.9611208134701578, 0.9193020930549909, 1.0673533629722003,
              0.7101556085362439],
        'eqp_lin': [-20.136940456159117, -1.3098066926937815, -0.6748384817370601,
                    -0.502978505295935, -0.4336239113743001, 0.1885056540147607, 0.282140486426722,
                    1.0269565116356945, 1.11914672538343, 1.1012151384283135, 1.1680617462891372,
                    1.2600983883942252, 1.6833400739266036],
    },
    'water/os_tda_s1_d5_dmp04': {
        'n_it_1b': 1,
        'n_it_2b': [33],
        'err_1b': [0.5009176145881931],
        'eqp': [-20.05951610166711, -1.2988117676299606, -0.674752434294608, -0.5026815142668893,
                -0.433132825682752, 0.1885021728073561, 0.28212986972196874, 1.0270078610038524,
                1.1189335188421197, 1.1164520994817353, 1.1671072702425087, 1.2810947099408185,
                1.6768587649438618],
        'sigc': [0.6984641329772813, 0.05591327397709272, 0.03723423798038171, 0.06209011998221786,
                 0.0738196201937607, -0.01548234073367279, -0.018136320975812723,
                 -0.03174251816213362, -0.04913879160456113, -0.09011187186462855,
                 -0.052987036329140175, -0.14462327226312388, -0.01739902594571939],
        'z': [0.7983114432311381, 0.768231654400725, 0.9508523372938337, 0.9371983400346579,
              0.9303706981924504, 0.9834039016751671, 0.9759378637385361, 0.9545989643860042,
              0.9328668478261608, 0.961120801825509, 0.9193021082453947, 1.067353390425792,
              0.7101556751519856],
        'eqp_lin': [-20.13694048441581, -1.3098066921211378, -0.6748384707742127,
                    -0.5029785082208917, -0.43362392160372576, 0.18850564740309755,
                    0.2821404796670992, 1.0269565114259651, 1.1191467370154649, 1.1012151420385834,
                    1.1680617564581164, 1.260098398488118, 1.6833400720759895],
    },
    'water/ev_n50_dmp04': {
        'n_it_1b': 6,
        'n_it_2b': [81, 66, 61, 49, 44, 36],
        'err_1b': [0.5009176791613719, 0.1455086668289347, 0.05663037558796802,
                   0.023891014728377513, 0.010383417872006362, 0.004569580124350381],
        'eqp': [-19.818532981952295, -1.289470281522289, -0.6688600365345291, -0.49299958639051256,
                -0.4204015340540209, 0.18692390768978123, 0.2794666761080801, 1.0198201314998787,
                1.10938320970794, 1.0934514103545203, 1.155624506099932, 1.2432257152258246,
                1.6772885929056924],
        'sigc': [0.7455325741816298, 0.06680628026041298, 0.0412177768176099, 0.06760250195658316,
                 0.08097082109476055, -0.016800285880937547, -0.02035295854066821,
                 -0.03771725122651105, -0.055074062233954044, -0.09374871186967236,
                 -0.059941526770406005, -0.13628323089547575, -0.018810541782161627],
        'z': [0.5591404598664678, 0.8908400823674146, 0.9289019318376436, 0.91180492341245,
              0.8976278361757789, 0.9797418019853389, 0.9674876713254399, 0.9632345140209816,
              0.893467858604385, 0.9006538177315065, 0.8413146267060265, 0.8608165028974977,
              0.9202382027896815],
        'eqp_lin': [-20.14654773893849, -1.2967545039148445, -0.671790241182157,
                    -0.49896113966046496, -0.42868959675632234, 0.18726422156738526,
                    0.2801283176940553, 1.0212062649576314, 1.1152494251484897, 1.1027593571167154,
                    1.1651358454206007, 1.2621863664875186, 1.6787794785801973],
    },
    'water/ev_l50_dmp04': {
        'n_it_1b': 6,
        'n_it_2b': [81, 66, 53, 41, 27, 20],
        'err_1b': [0.42349325966437945, 0.012130989297512107, 0.002710464500023768,
                   0.0004117379563883805, 0.00011019379535692764, 2.2816753640242382e-05],
        'eqp': [-20.145747423903707, -1.2978519000057638, -0.6720539912429795,
                -0.49931868898770715, -0.4291438028770181, 0.18738469885671596, 0.2803387304784403,
                1.0218051185927124, 1.115577312302696, 1.102052158210087, 1.1652511726890855,
                1.2605477927300361, 1.6805060686344502],
        'sigc': [0.741307041127081, 0.06531804776228502, 0.04083954994129028, 0.06709841987584844,
                 0.08029279761719225, -0.016672124207412403, -0.02012046980251047,
                 -0.03705562651797201, -0.05451863413414739, -0.09379722531326072,
                 -0.05954582117377919, -0.14020382011439797, -0.01733472369679321],
        'z': [0.5593988311794639, 0.8943825629299278, 0.9310452672833667, 0.9133239198745717,
              0.8995467078415428, 0.9800469244701102, 0.9682089006124771, 0.9642727622141145,
              0.896550933289206, 0.9077101749624658, 0.8449605126715194, 0.8483370343212581,
              0.899053932925968],
        'eqp_lin': [-20.145747423903707, -1.2978519000057638, -0.6720539912429795,
                    -0.49931868898770715, -0.4291438028770181, 0.18738469885671596,
                    0.2803387304784403, 1.0218051185927124, 1.115577312302696, 1.102052158210087,
                    1.1652511726890855, 1.2605477927300361, 1.6805060686344502],
    },
    'ammonia/os_tda_s1_dmp04': {
        'n_it_1b': 1,
        'n_it_2b': [60],
        'err_1b': [0.4402217771335888],
        'eqp': [-15.099532760836881, -1.0879070581657986, -0.5874324039917967, -0.5874112230937372,
                -0.3591577727854062, 0.19593321677377945, 0.2968223965298426, 0.296827352452084,
                0.8994801189562225, 0.8995041707842202, 0.9243985237398673, 1.148189086113337,
                1.1721399754558908, 1.1721589984145284, 1.4667689890016595],
        'sigc': [0.6266840846708456, 0.05930695891525049, 0.039252453524372496,
                 0.03924971451847134, 0.06026354619753349, -0.019926707578195427,
                 -0.02284425468289314, -0.02284400209294819, -0.02738754679128581,
                 -0.027387946787919473, -0.05907182732401818, -0.1210819208506304,
                 -0.11807898594529657, -0.11809712473885021, -0.0399270281635266],
        'z': [0.7870409724617942, 0.7057500792324086, 0.9445435221949893, 0.9445458901215853,
              0.9321051248627271, 0.9780325178952073, 0.9694285357598555, 0.9694286018003017,
              0.961719308602358, 0.961721454570192, 0.9152947240687804, 0.9798187701749189,
              1.041822241521118, 1.0418958973068182, 0.6701109587292404],
        'eqp_lin': [-15.168380586087451, -1.1056796649381202, -0.5875489361941503,
                    -0.5875277287677468, -0.3594591297869223, 0.19594248357246113,
                    0.2968459779846476, 0.2968509338613502, 0.8993750112701498, 0.8993988328859399,
                    0.9255693024759222, 1.141434796453106, 1.1068383600119474, 1.1068399461209406,
                    1.4887028203357822],
    },
    'ammonia/os_tda_s1_d5_dmp04': {
        'n_it_1b': 1,
        'n_it_2b': [31],
        'err_1b': [0.44022174037042205],
        'eqp': [-15.099532797600048, -1.0879070557193722, -0.5874323968540434, -0.5874112159561775,
                -0.3591577722285694, 0.19593321355643414, 0.2968223934223409, 0.29682734934491184,
                0.8994801204955457, 0.899504172323058, 0.9243985276980364, 1.148189085730929,
                1.1721399736242062, 1.1721589965826407, 1.4667689894461045],
        'sigc': [0.6266839991311693, 0.05930696073031685, 0.039252460726636726,
                 0.03924972172051672, 0.06026354620248201, -0.01992671080082759,
                 -0.022844257786057347, -0.022844005195776396, -0.0273875452279894,
                 -0.027387945225122574, -0.05907182241683616, -0.12108191291873348,
                 -0.11807898216224105, -0.11809712095324444, -0.03992702831443441],
        'z': [0.7870409931822102, 0.7057501150439431, 0.944543531506839, 0.9445458994338112,
              0.9321051336636675, 0.9780325213117969, 0.969428540296526, 0.9694286063367674,
              0.9617193130985449, 0.961721459065478, 0.9152947328360274, 0.9798187217523096,
              1.0418222204833638, 1.0418958762384554, 0.6701109845836988],
        'eqp_lin': [-15.16838060158104, -1.1056796612724336, -0.5875489290877857,
                    -0.5875277216615743, -0.35945912920585316, 0.195942480357788,
                    0.2968459748819008, 0.29685093075893304, 0.8993750127021957,
                    0.8993988343175514, 0.9255693062414839, 1.1414348028882466, 1.106838366641263,
                    1.1068399527560382, 1.488702819400483],
    },
    'ammonia/ev_n50_dmp04': {
        'n_it_1b': 6,
        'n_it_2b': [69, 59, 54, 46, 37, 29],
        'err_1b': [0.4402217741814276, 0.1378475371944159, 0.0570898259250896,
                   0.025333999661372175, 0.01153755502043552, 0.00531430759364504],
        'eqp': [-14.862409538394084, -1.0834071624067358, -0.5807171270475415, -0.5806963587362806,
                -0.3484134720413467, 0.1934645033769754, 0.29309955034981716, 0.29310448688175345,
                0.8925267263096126, 0.8925506654111881, 0.9131825300176835, 1.1177810252685847,
                1.1239031725944302, 1.1239187642740687, 1.4682939914037842],
        'sigc': [0.6819392244303037, 0.06504295727127023, 0.04369252298007137, 0.0436896345755811,
                 0.06663945523809962, -0.02194875195865359, -0.02584641772141226,
                 -0.02584619383692404, -0.03329491536826214, -0.033295225597220576,
                 -0.06538584241029756, -0.11782072475054815, -0.11414854994178973,
                 -0.11415680990760114, -0.04741210033300667],
        'z': [0.5373080612819163, 0.7475031859442084, 0.9198308755413548, 0.9198346814313775,
              0.9045180930872079, 0.9718279597113909, 0.9565229416072064, 0.9565228110279982,
              0.9609230340881216, 0.9609209666034693, 0.8620873327767088, 0.7815350264622785,
              0.9284556459262784, 0.9284842527369395, 0.692387403294997],
        'eqp_lin': [-15.176771806151269, -1.0998297717895047, -0.5842195791405016,
                    -0.584198412978234, -0.3547753795094368, 0.19408277132099647,
                    0.29422308447977547, 0.2942280145991082, 0.8938271250182757,
                    0.8938511444366306, 0.9221993681361185, 1.1435217531422166, 1.132031399166585,
                    1.1320442996829767, 1.482877745620107],
    },
    'ammonia/ev_l50_dmp04': {
        'n_it_1b': 6,
        'n_it_2b': [69, 62, 50, 50, 46, 46],
        'err_1b': [0.37137395084010194, 0.045248468868122727, 0.03591049281355807,
                   0.024189645495749534, 0.01741163793434719, 0.012369524844201507],
        'eqp': [-15.175823153039671, -1.1013108870223134, -0.584635261512741, -0.5846140521709018,
                -0.3552617295577296, 0.19430558318739405, 0.29446028305459054, 0.2944650631933351,
                0.8944395241983657, 0.8944621012180706, 0.9224571183523043, 1.1448107300106904,
                1.13531278970472, 1.1353254597846434, 1.4852522966986268],
        'sigc': [0.6765371418626265, 0.06267959645477768, 0.043131833378151796,
                 0.04312900311726692, 0.06596526384720018, -0.021706135047454157,
                 -0.02556780548762698, -0.025567744766870997, -0.0325952516723544,
                 -0.0325971547279466, -0.06479517366204293, -0.12244060692635368,
                 -0.10608160208507329, -0.10609010717980201, -0.0436386588707163],
        'z': [0.5379326017915753, 0.7520602397181987, 0.9221493081471273, 0.9221530767601076,
              0.906386686317324, 0.9724252955690126, 0.9576681790412019, 0.9576678113619282,
              0.9627607309902343, 0.9627558054480657, 0.865962209256393, 0.7414693121858498,
              0.9680603758808949, 0.9680879398936347, 0.6978449245943935],
        'eqp_lin': [-15.175823153039671, -1.1013108870223134, -0.584635261512741,
                    -0.5846140521709018, -0.3552617295577296, 0.19430558318739405,
                    0.29446028305459054, 0.2944650631933351, 0.8944395241983657,
                    0.8944621012180706, 0.9224571183523043, 1.1448107300106904, 1.13531278970472,
                    1.1353254597846434, 1.4852522966986268],
    },
    'water/qs_tda_s1_dmp04': {
        'n_it_1b': 4,
        'n_it_2b': [68, 72, 72, 72],
        'err_1b': [0.01341966009664064, 0.008158518482701105, 0.0024600130076680715,
                   0.00013922831747148479],
        'eqp': [-19.960561148532697, -1.2906898519081134, -0.6830689492836881, -0.509871012382924,
                -0.4409235894113181, 0.188938857240302, 0.28337036653717235, 1.0253194424454721,
                1.122886674439612, 1.1239211986044404, 1.1799683374193395, 1.3068182857166744,
                1.6524585635490157],
        'sigc': [0.6121377532294349, 0.07018167888169055, 0.032055804655767656,
                 0.056473569149854644, 0.06525449621799626, -0.013660183209601547,
                 -0.015126141253471908, -0.031165728078347065, -0.06112751596680507,
                 -0.03669375922704505, -0.03140253630917039, -0.06495989051069853,
                 -0.039232096357969454],
    },
    'ammonia/qs_tda_s1_dmp04': {
        'n_it_1b': 4,
        'n_it_2b': [60, 64, 64, 64],
        'err_1b': [0.006252164350119327, 0.002307683471985067, 0.0006471694762725244,
                   0.00010200041845775942],
        'eqp': [-15.034034472884086, -1.0913649660300377, -0.595288894046251, -0.5952675201419574,
                -0.36646048821694455, 0.19788048398625788, 0.29977260113234344,
                0.29977776657500327, 0.9002989447145462, 0.9003234214748906, 0.9353724460631989,
                1.1613820401709785, 1.1614035287540023, 1.1776042192283809, 1.4776961688644823],
        'sigc': [0.5078294763738161, 0.05884980011893926, 0.03160722245271933, 0.03160504280810603,
                 0.05058779864706179, -0.016869072590409533, -0.018650343000689483,
                 -0.01864998357658448, -0.02370245251868618, -0.023702202310209752,
                 -0.042154532611561074, -0.07357864561030514, -0.07358111136292868,
                 -0.05600203617149361, -0.035595275303591516],
    },
}


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def run(driver, scheme, **options):
    """One run of water/6-31G through `driver`'s lower level."""
    if driver == 'restricted':
        if scheme == 'qs':
            return quiet(restricted.qs_parquet, **qs_inputs('water'), **options)
        eps, eri, nO = mo_integrals('water')
        return quiet(restricted.ev_parquet, eps, eri, nO, scheme, **options)
    if scheme == 'qs':
        ref = spin_orbital.spin_orbital_reference(rhf('water'))
        x = spin_orbital.orthogonalization_matrix(ref.ovlp)
        return quiet(spin_orbital.qs_parquet_so, ref.eps, ref.mo_coeff, ref.ovlp, x, ref.hcore,
                     ref.eri_ao, ref.nO, **options)
    eps, eri, nO, pairs = both_layouts('water')[1]
    return quiet(spin_orbital.ev_parquet_so, eps, eri, nO, scheme, spin_pairs=pairs, **options)


def default_builder(driver):
    return static_reducible_vertex if driver == 'restricted' else static_reducible_vertex_so


def assert_same_run(got, want, label):
    """Two runs of one computation give the same numbers: eqp, z and the last
    vertices as `tests.pins.assert_same` holds two computations (to a few ulp
    under the recording BLAS, within the class bar elsewhere: two runs of one
    code need not round alike), the counts and the flags raised exactly."""
    assert got.n_it_2b == want.n_it_2b, (label, got.n_it_2b, want.n_it_2b)
    assert_same(got.eqp, want.eqp, label=(label, 'eqp'))
    assert_same(got.z, want.z, unit='1', label=(label, 'z'))
    for name, x, y in zip(got.vertices._fields, got.vertices, want.vertices):
        assert_same(x, y, quantity='root', label=(label, name))
    assert ([e['flag'] for e in got.numerical_flags]
            == [e['flag'] for e in want.numerical_flags]), label


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('scheme', sorted(SCHEMES))
@pytest.mark.parametrize('max_diis_2b', [1, 3])
def test_quacks_vertex_through_the_hook_is_the_default_run(driver, scheme, max_diis_2b):
    name, options = SCHEMES[scheme]
    default = default_builder(driver)
    reference = run(driver, name, max_diis_2b=max_diis_2b, **options)

    def own(vertex_name, spectra, eta_2b):
        return default(vertex_name, spectra, eta_2b)
    partial = functools.partial(default)
    for builder, label in ((own, 'own'), (partial, f'partial({default.__name__})')):
        result = run(driver, name, max_diis_2b=max_diis_2b, vertex_builder=builder, **options)
        assert_same_run(result, reference, (driver, scheme, label))
        assert result.options['vertex_builder'] == label
    assert reference.options['vertex_builder'] == default.__name__


class Keeper:
    """A builder that caches what it hands out, with a copy of each array as
    it left (the loop must only read them)."""

    def __init__(self, default):
        self.default = default
        self.handed, self.copies = [], []

    def __call__(self, name, spectra, eta_2b):
        phi = self.default(name, spectra, eta_2b)
        self.handed.append(phi)
        self.copies.append(phi.copy())
        return phi


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('max_diis_2b', [1, 3])
@pytest.mark.parametrize('damping', [None, DAMPING])
def test_a_cached_vertex_is_only_read(driver, max_diis_2b, damping):
    """At the default damping and at 0.4: the spin-orbital default (alpha
    = 1) mixes 1 Phi_new + 0 Phi_old, which leaves the builder's array as it
    was even if the loop wrote into it."""
    keeper = Keeper(default_builder(driver))
    options = dict(SCHEMES['ev'][1], max_diis_2b=max_diis_2b, damping=damping)
    result = run(driver, 'ev', vertex_builder=keeper, **options)
    assert keeper.handed and all(a.tobytes() == b.tobytes()
                                 for a, b in zip(keeper.handed, keeper.copies))
    assert_same_run(result, run(driver, 'ev', **options), (driver, max_diis_2b, damping))
    assert result.options['vertex_builder'] == f'{Keeper.__module__}.{Keeper.__qualname__}'


#: A vertex element no kernel and no effective integral reads, per driver:
#: the closed-shell triplet pp pair (i, j) = (0, 0) (triplet pairs have
#: i < j) and the spin-orbital eh pair p = q.
UNREAD_ELEMENT = {'restricted': ('pp_triplet', (0, 0, 0, 0)), 'spin_orbital': ('eh', (0, 0, 0, 0))}


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_a_nan_the_kernels_never_read(driver):
    """A NaN at `UNREAD_ELEMENT`: kept and flagged; the change of that vertex
    is the finite maximum over its other elements (gfortran's maxval passes
    over NaN), so the loop runs on, and without DIIS the run is the default
    one."""
    default = default_builder(driver)
    vertex, index = UNREAD_ELEMENT[driver]

    def build(name, spectra, eta_2b):
        phi = default(name, spectra, eta_2b)
        if name == vertex:
            phi = phi.copy()
            phi[index] = np.nan
        return phi
    options = dict(eta_2b=1.0, max_it_2b=6, conv_2b=1e-12, **NEWTON)
    result = run(driver, 'os', vertex_builder=build, **options)
    reference = run(driver, 'os', **options)
    assert result.n_it_2b == reference.n_it_2b == [6]
    assert_same(result.eqp, reference.eqp, label=(driver, 'eqp'))
    names = {e['flag'] for e in result.numerical_flags}
    assert 'parquet_vertex_not_finite' in names and 'parquet_two_body_not_finite' not in names
    k = result.vertices._fields.index(vertex)
    for record in result.history[0]['two_body']:
        assert np.isfinite(record['err_2b']) and np.isfinite(record['err'][k]), record
    assert np.isnan(getattr(result.vertices, vertex)[index])


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_the_recorded_rcond_is_that_of_the_damped_errors(driver):
    """The history's two-body rcond at the second iteration equals that of
    Quack's DIIS fed Quack's pairs: Phi = alpha new + (1 - alpha) old and
    error Phi - old, the old vertices zero at the start."""
    keeper = Keeper(default_builder(driver))
    alpha = 1.0 - DAMPING
    result = run(driver, 'os', eta_2b=1.0, max_it_2b=2, conv_2b=1e-12, max_diis_2b=3,
                 damping=DAMPING, vertex_builder=keeper, qp_solver='linearized')
    k = len(result.vertices)
    new1, new2 = keeper.copies[:k], keeper.copies[k:2 * k]
    damped1 = [alpha * n + (1.0 - alpha) * np.zeros_like(n) for n in new1]
    damped2 = [alpha * n + (1.0 - alpha) * d for n, d in zip(new2, damped1)]
    reference = QuackDIIS(3, [n.shape for n in new1])
    reference.push(damped1, damped1)
    reference.push([d2 - d1 for d2, d1 in zip(damped2, damped1)], damped2)
    got = result.history[0]['two_body'][1]['diis_rcond']
    assert got == pytest.approx(reference.rcond, rel=RCOND_RTOL), (got, reference.rcond)


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
@pytest.mark.parametrize('scheme', sorted(SCHEMES))
def test_the_default_damping_is_quacks_alpha(driver, scheme):
    """None (and no argument) is the explicit default, the same run: 0.75
    for the closed-shell driver and qsPA (Quack's alpha 0.25), 0 for the
    spin-orbital FLEX, osPA and evPA (G_evParquet's alpha 1), recorded."""
    name, options = SCHEMES[scheme]
    want = (PARQUET_SO_EV_VERTEX_DAMPING if driver == 'spin_orbital' and scheme != 'qs'
            else PARQUET_VERTEX_DAMPING)
    implicit = run(driver, name, **options)
    explicit = run(driver, name, damping=want, **options)
    none = run(driver, name, damping=None, **options)
    assert_same_run(implicit, explicit, (driver, scheme, 'no damping given'))
    assert_same_run(none, explicit, (driver, scheme, 'damping None'))
    assert implicit.options['damping'] == want and isinstance(implicit.options['damping'], float)


@pytest.mark.parametrize('driver', ['restricted', 'spin_orbital'])
def test_the_mixing_is_quacks_to_the_bit(driver):
    """FLEX's vertices are alpha times the builder's (damping 0.75 and 0);
    two osPA iterations give fl(fl(alpha Phi_2) + fl((1 - alpha) alpha Phi_1))."""
    default = default_builder(driver)
    for damping in (PARQUET_VERTEX_DAMPING, 0.0, DAMPING):
        keeper = Keeper(default)
        result = run(driver, 'flex', eta_2b=1.0, damping=damping, vertex_builder=keeper,
                     **NEWTON)
        alpha = 1.0 - damping
        for got, new in zip(result.vertices, keeper.copies):
            assert np.array_equal(got, alpha * new + (1.0 - alpha) * np.zeros_like(new)), damping
        keeper = Keeper(default)
        result = run(driver, 'os', eta_2b=1.0, max_it_2b=2, conv_2b=1e-12, damping=damping,
                     vertex_builder=keeper, **NEWTON)
        k = len(result.vertices)
        for got, first, second in zip(result.vertices, keeper.copies[:k], keeper.copies[k:]):
            old = alpha * first + (1.0 - alpha) * np.zeros_like(first)
            assert np.array_equal(got, alpha * second + (1.0 - alpha) * old), damping


@pytest.mark.parametrize('case', sorted(DAMPED))
def test_damping_against_quack(case):
    """Damping 0.4 against Quack compiled with alpha = 0.6, at the bars of
    `test_parquet_restricted_vs_quack` (port and Quack agree to 1e-13 Ha)."""
    result = run_case(case, DAMPED)
    bar_class = DAMPED[case][3]
    assert_matches_quack(result, case, QUACK_DAMPED[case], *bars(bar_class))
    assert result.options['damping'] == DAMPING


def test_every_damped_case_has_its_quack_numbers():
    assert set(DAMPED) == set(QUACK_DAMPED)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-q']))
