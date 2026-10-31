"""Live W&B query table: rows derive from summary metrics, never frozen snapshots.

Uses the same query-panel node protocol as wandb_workspaces.WeavePanel.
No training-side logger, artifact writer, polling service or training restart is needed.
"""
from .course_config import COURSES,SUCCESS
from .parallel_curriculum import PREREQUISITES

COLUMNS=('课程','当前成功率','稳定率','状态')
LABELS=('站立','扰动站立','重心转移','原地踏步','慢走','前进','后退','侧移',
        '速度跟踪','转向','行走转向','停步','启停','接近球','对齐','左脚正踢','右脚正踢',
        '左脚侧踢','右脚侧踢','踢后恢复','盘带','传接球','守门','1v1','2v1','2v2','团队配合','4v4比赛')
MAYBE_NUMBER={'type':'union','members':['none','number']}


def const(value):
    return {'nodeType':'const','type':'boolean' if isinstance(value,bool) else 'number' if isinstance(value,(int,float)) else 'string','val':value}


def op(name,kind,**inputs):
    return {'nodeType':'output','type':kind,'fromOp':{'name':name,'inputs':inputs}}


def record(fields):
    return op('dict',{'type':'typedDict','propertyTypes':{k:v['type'] for k,v in fields.items()}},**fields)


def pick(obj,key,kind=MAYBE_NUMBER):
    return op('pick',kind,obj=obj,key=const(key))


def compare(name,lhs,rhs):return op('number-'+name,'boolean',lhs=lhs,rhs=const(rhs))


def all_of(values):
    result=const(True)
    for value in values:result=op('and','boolean',lhs=result,rhs=value)
    return result


def choose(cases,default):
    cases=[*cases,(const(True),const(default))]
    return op('cond','string',cases=record({str(i):a for i,(a,_) in enumerate(cases)}),
              results=record({str(i):b for i,(_,b) in enumerate(cases)}))


def percent(value):
    rounded=op('number-toFixed',MAYBE_NUMBER,**{'in':op('number-mult',MAYBE_NUMBER,lhs=value,rhs=const(100)),'digits':const(2)})
    formatted=op('string-add','string',lhs=op('number-toString','string',**{'in':rounded}),rhs=const('%'))
    return choose([(compare('greaterEqual',value,0),formatted)],'—')


def course_table_config(parallel=True):
    runs={'nodeType':'var','type':{'type':'list','objectType':'run'},'varName':'runs'}
    run=op('index','run',arr=runs,index=const(0))
    summary=op('run-summary',{'type':'typedDict','propertyTypes':{}},run=run)
    metric=lambda name,key:pick(summary,'06_课程/'+name+'/'+key)
    current=pick(summary,'01_总览/课程编号')
    rows=[]
    for i,name in enumerate(COURSES):
        success=metric(name,'成功率');stable=metric(name,'稳定率')
        if parallel:
            known=all_of([compare('greaterEqual',metric(name,key),0) for key in
                          ('红队成功率','蓝队成功率','红队稳定率','蓝队稳定率')])
            passed=all_of([compare('greaterEqual',metric(name,key),target) for key,target in
                           (('红队成功率',SUCCESS[name]),('蓝队成功率',SUCCESS[name]),('红队稳定率',.9),('蓝队稳定率',.9))])
            review=choose([(known,choose([(passed,const('复习·达标'))],'复习·退化'))],'复习·待采样')
            ready=all_of([compare('equal',metric(parent,'已掌握'),1) for parent in PREREQUISITES[name]])
            status=choose([(compare('equal',metric(name,'已掌握'),1),review),
                           (ready,choose([(all_of([known,passed]),const('主训·达标待认证'))],'主训')),
                           (compare('greater',metric(name,'球场数'),0),const('预习'))],'未解锁')
            success_text=percent(success);stable_text=percent(stable)
        else:
            active=compare('equal',current,i)
            status=choose([(active,const('主训')),(compare('greater',current,i),const('已通过·无复测'))],'待训练')
            success_text=choose([(active,percent(success))],'—')
            stable_text=choose([(active,percent(stable))],'—')
        rows.append(record(dict(zip(COLUMNS,(const(f'{i} {LABELS[i]}'),success_text,stable_text,status)))))
    row_type=rows[0]['type']
    expression=op('list',{'type':'list','objectType':row_type},**{str(i):r for i,r in enumerate(rows)})
    row={'nodeType':'var','type':row_type,'varName':'row'}
    table_state={'autoColumns':False,'columns':{str(i):{'panelId':''} for i in range(4)},
                 'columnNames':{str(i):name for i,name in enumerate(COLUMNS)},
                 'columnSelectFunctions':{str(i):pick(row,name,'string') for i,name in enumerate(COLUMNS)},
                 'preFilterFunction':{'nodeType':'void','type':'invalid'},'order':[str(i) for i in range(4)],
                 'groupBy':[],'sort':[],'pageSize':28,'page':0}
    return {'panel2Config':{'exp':dict(expression,__userInput=True),'panelId':'table',
            'panelConfig':{'tableState':table_state,'rowSize':1,'indexOffset':0,
                           'columnWidths':{'0':210,'1':150,'2':150,'3':230},'pinnedRows':{},'pinnedColumns':[],
                           'activeRowForGrouping':{},'simpleTable':True}}}
