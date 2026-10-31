from q1_4v4.course_table import course_table_config,COLUMNS


def evaluate(node,summary):
    """Exercise query bindings with changing summaries without contacting W&B."""
    if node['nodeType']=='const':return node['val']
    if node['nodeType']=='var':return [{'summary':summary}]
    name=node['fromOp']['name'];v={k:evaluate(x,summary) for k,x in node['fromOp']['inputs'].items()}
    if name=='dict':return v
    if name=='list':return list(v.values())
    if name=='index':return v['arr'][int(v['index'])]
    if name=='run-summary':return v['run']['summary']
    if name=='pick':return v['obj'].get(v['key'])
    if name=='and':return None if v['lhs'] is None or v['rhs'] is None else v['lhs'] and v['rhs']
    if name=='cond':return next((v['results'][k] for k,c in v['cases'].items() if c),None)
    if name=='string-add':return (v['lhs'] or '')+(v['rhs'] or '')
    if name=='number-toFixed':return None if v['in'] is None else round(v['in'],int(v['digits']))
    if name=='number-toString':return None if v['in'] is None else format(v['in'],'g')
    a,b=v['lhs'],v['rhs']
    if a is None or b is None:return None
    return {'number-equal':lambda:a==b,'number-greater':lambda:a>b,
            'number-greaterEqual':lambda:a>=b,'number-mult':lambda:a*b}[name]()


def sample(summary,name,success=1.,stable=1.,mastered=0,worlds=100):
    for key,value in {'成功率':success,'稳定率':stable,'红队成功率':success,'蓝队成功率':success,
                      '红队稳定率':stable,'蓝队稳定率':stable,'已掌握':mastered,'球场数':worlds}.items():
        summary[f'06_课程/{name}/{key}']=value


def test_live_table_tracks_new_values_and_identifies_regression():
    summary={}
    for name in ('balance','balance_push','weight_shift'):sample(summary,name,mastered=1)
    sample(summary,'step_in_place',success=0.)
    sample(summary,'slow_walk',success=0.)
    expr=course_table_config()['panel2Config']['exp']
    rows=evaluate(expr,summary)
    assert list(rows[0])==list(COLUMNS) and len(rows)==28
    assert rows[0]['状态']=='复习·达标'
    assert rows[3]['状态']=='主训' and rows[3]['当前成功率']=='0%'
    assert rows[4]['状态']=='预习'
    assert rows[5]['状态']=='未解锁' and rows[5]['当前成功率']=='—'
    # A team can fail even when the overall mean appears to pass.
    summary['06_课程/weight_shift/蓝队成功率']=.6
    summary['06_课程/weight_shift/成功率']=.85
    changed=evaluate(expr,summary)
    assert changed[2]['当前成功率']=='85%' and changed[2]['状态']=='复习·退化'
    sample(summary,'step_in_place',success=.9)
    assert evaluate(expr,summary)[3]['状态']=='主训·达标待认证'
    sample(summary,'step_in_place',success=.9,mastered=1)
    assert evaluate(expr,summary)[4]['状态']=='主训'


def test_missing_team_is_waiting_for_evidence_not_regression():
    summary={};sample(summary,'balance',mastered=1)
    summary['06_课程/balance/蓝队成功率']=None
    rows=evaluate(course_table_config()['panel2Config']['exp'],summary)
    assert rows[0]['状态']=='复习·待采样'


def test_serial_mode_does_not_present_old_values_as_current_measurements():
    summary={'01_总览/课程编号':3}
    sample(summary,'balance',mastered=1)
    sample(summary,'step_in_place',success=0.)
    rows=evaluate(course_table_config(parallel=False)['panel2Config']['exp'],summary)
    assert rows[0]['状态']=='已通过·无复测' and rows[0]['当前成功率']=='—'
    assert rows[3]['状态']=='主训' and rows[3]['当前成功率']=='0%'
    assert rows[4]['状态']=='待训练'


def test_wandb_query_panel_roundtrip_preserves_live_expression():
    from wandb_workspaces.reports.v2 import WeavePanel,Layout
    config=course_table_config()
    panel=WeavePanel(config=config,layout=Layout(w=24,h=18))
    assert WeavePanel._from_model(panel._to_model()).config==config
    state=config['panel2Config']['panelConfig']['tableState']
    assert list(state['columnNames'].values())==list(COLUMNS)
    assert len(state['columns'])==4 and state['pageSize']==28
