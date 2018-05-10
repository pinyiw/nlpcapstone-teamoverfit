import matplotlib.pyplot as plt

l1 = \
[(0.17821937799453735, 2.68888336424278), (-0.4600532352924347, -0.03636127524187494), (-0.22034570574760437, 0.10002987905479344), (-0.06332322955131531, 1.517303264130159), (1.1025767773389816, 0.06264130819889213), (0.5119353532791138, -0.5097599036329276), (-0.4163600504398346, 0.5033827910594519), (-0.06716996431350708, -0.1967670336730903), (-0.8017562329769135, -0.09857748482930682), (0.11371821165084839, -0.8434037115621471), (0.1455247402191162, -0.9319057958170887), (0.12119784951210022, 0.3744983315768511), (0.34859515726566315, -0.7189088627054887), (0.05914047360420227, 0.7699327875045279), (0.1215573400259018, 0.9822043319061996), (0.14638081192970276, 0.9817479782001197), (-1.2046344578266144, 1.6321988230551623), (0.29876790940761566, -0.5704262931513064), (0.7879339158535004, 1.668147964049053), (0.3652304410934448, 0.0), (0.14199092984199524, 0.5469257899202619), (0.29202811419963837, 0.129491117525967), (-0.2668328583240509, 0.5777336337395744), (0.5666971206665039, 0.2657336540564141), (0.21466165781021118, 0.09404268386304092), (0.04711821675300598, -0.6577674429955217), (0.16277432441711426, 0.1978384287329926), (0.886453315615654, 0.6350726835563982), (0.830429419875145, -0.4264234634703598), (-0.7813133299350739, -0.025689734417074964), (0.0889994204044342, -0.7795429198728877), (-0.2429664134979248, 0.2848804585571224), (0.44370144605636597, -0.11190683448593444), (0.45470520853996277, 0.5085417961910798), (-0.1474626362323761, 1.1148285673289329), (1.19456946849823, 0.9158967135012738), (0.1754969358444214, 0.1009189021875595), (0.13571381568908691, 0.5373203364569159), (0.037063658237457275, -0.4175566334081075), (-1.6469307243824005, -0.17615876362043456), (0.861966609954834, 0.8064979220465356), (-0.5561232566833496, -0.008332036766386249), (0.4822254180908203, -0.17498735210466132), (0.3412950783967972, 0.18364143243725156), (-0.5047254264354706, 0.06665629413110206), (-0.4933066666126251, -0.09159135302859948), (0.21932199597358704, 1.592072040813379), (0.6504155695438385, 0.04922108470547297), (-0.5423344671726227, 0.00819947824544425), (0.44122226536273956, -0.2624454528418042), (0.34552738070487976, -0.2301706399901355), (0.17255470156669617, 6.098039792671178), (-0.09781196713447571, 1.196168457287398), (0.7087588310241699, 0.9517313852883365)]
l2 = \
[('UP', 'UP'), ('STAY', 'STAY'), ('UP', 'STAY'), ('UP', 'UP'), ('STAY', 'STAY'), ('DOWN', 'DOWN'), ('UP', 'UP'), ('STAY', 'STAY'), ('UP', 'STAY'), ('STAY', 'DOWN'), ('UP', 'DOWN'), ('STAY', 'STAY'), ('UP', 'DOWN'), ('DOWN', 'UP'), ('UP', 'UP'), ('UP', 'UP'), ('UP', 'UP'), ('STAY', 'DOWN'), ('STAY', 'UP'), ('UP', 'STAY'), ('UP', 'UP'), ('STAY', 'STAY'), ('UP', 'UP'), ('UP', 'STAY'), ('UP', 'STAY'), ('STAY', 'DOWN'), ('STAY', 'STAY'), ('UP', 'UP'), ('STAY', 'STAY'), ('STAY', 'STAY'), ('UP', 'DOWN'), ('STAY', 'STAY'), ('STAY', 'STAY'), ('UP', 'UP'), ('UP', 'UP'), ('UP', 'UP'), ('UP', 'STAY'), ('UP', 'UP'), ('UP', 'STAY'), ('STAY','STAY'), ('STAY', 'UP'), ('UP', 'STAY'), ('UP', 'STAY'), ('UP', 'STAY'), ('STAY', 'STAY'), ('UP', 'STAY'), ('STAY', 'UP'), ('STAY', 'STAY'), ('UP', 'STAY'), ('STAY', 'STAY'), ('DOWN', 'STAY'), ('STAY', 'UP'), ('UP', 'UP'), ('UP', 'UP')]
plt.plot(list(range(len(l1))) ,[l[0] for l in l1], label='prediction')
plt.plot(list(range(len(l1))) ,[l[1] for l in l1], label='target')
plt.legend()
plt.show()
# plt.savefig('plots/55accuracy.png')

class2idx = {'DOWN': -1, 'STAY': 0, 'UP': 1}
plt.scatter(list(range(len(l2))), [class2idx[l[0]] for l in l2], label='prediction')
plt.scatter(list(range(len(l2))), [class2idx[l[1]] for l in l2], label='target')

plt.legend()
# plt.show()
