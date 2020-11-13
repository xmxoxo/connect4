#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'


import os
import sys

import logging
import numpy as np
import json


sys.setrecursionlimit(2000)
# In[399]:


class Board:    
    def __init__(self, N):
        '''Board for a Connect-N game, the board will have N+2 Col and N+3 Rows'''
        self.N = N
        self.num_row = N + 3
        self.num_col = N + 2        
        self.cols = [Column(self.num_row) for i in range(self.num_col)] # create N+2 columns
        self.player = 1
        self.icon = ['\u25CB','\u25CF','\u25B2']
        self.winner = 0
        self.players = ['玩家', '电脑']
        self.allsteps = []
        # 获胜局数： 人类, 电脑
        self.score = np.loadtxt ('score.npy').astype(int)

    #更新获胜分数
    def update_score (self):
        if self.winner==1:
            self.score[0]+=1
        if self.winner==2:
            self.score[1]+=1
        if self.winner==3:
            self.score+=1

        np.savetxt('score.npy', self.score, fmt='%d')

        
    def getjson (self):
        js = {}
        js['N'] = self.N
        js['num_row'] = self.num_row
        js['num_col'] = self.num_col
        js['player'] = self.player
        js['winner'] = self.winner
        js['allsteps'] = self.allsteps
        js['status'] = self.status().tolist()
        return js
    
    def setjson (self, js):
        print(js)
        self.N = js['N']
        self.num_row = js['num_row']
        self.num_col = js['num_col']
        self.player = js['player']
        self.winner = js['winner']
        self.allsteps = js['allsteps']

        st = js['status']
        st = np.array(st).T[:,::-1]
        #print('setjson st:', st)

        self.cols = [Column(self.num_row) for i in range(self.num_col)]
        for i in range(self.num_col) :
            self.cols[i].setdata(st[i].tolist())
        #print('status:', self.status())

    def status(self):
        bd = [x.getdata() for x in self.cols]
        bd = np.array(bd)[:,::-1].T
        return bd
    
    def display(self):
        '''display the board'''

        bd = self.status()

        txt = ''
        print()
        print('重力四子棋游戏'.center(30,'-'))

        txt += ''.join(map(str, range(1,7))) + '\n'
        txtbd = ('\n').join([''.join(map(str, line))  for line in bd.tolist()])
        txt += txtbd.replace('0',self.icon[0]).replace('1',self.icon[1]).replace('2',self.icon[2])
        txt += '\n'
        txt += ''.join(map(str, range(1,7))) + '\n\n'
        txt += self.showtip() #+ '\n'
        txt += self.showplayer()
        print(txt)
        return txt
        
    def drop_disk(self, c):
        '''drop a disk at column c'''
        pass
        if c <1 or c >self.num_col :
            print('你下错了，请输入范围:1-%d' % self.num_col)
            return False
        ret = self.cols[c-1].drop_disk(self.player)
        if ret:
            self.allsteps.append(str(c))
            # 自动判断获胜情况
            if self.check_winning_condition():
                self.winner = self.player
                #更新得分
                self.update_score()

                logging.info('\n棋盘状态:\n%s\n获胜者：%s%s，对局:%s ' % 
                    (self.display(), self.icon[self.winner], self.players[self.winner-1], self.allsteps ))
            else:
                # 和棋的判断
                if len(self.allsteps) == self.num_row * self.num_col:
                    self.winner = 3
                    #更新得分
                    self.update_score()
                    logging.info('\n棋盘状态:\n%s\n双方和棋，对局:%s ' % (self.display(), self.allsteps ))
                else:
                    # 自动切换玩家
                    self.changeplayer()
        return ret

    def check_winning_condition(self):
        '''check if there is a winner'''
        ret = False
        # 每个点沿右，右下，下，左下4个方向最多走3步，每一步都必须与当前点同色，如果能走到3步则表示获胜
        dire = [[0,1],[1,1],[1,0],[1,-1]]
        bd = self.status()
        for x in range(bd.shape[0]):
            for y in range(bd.shape[1]):
                po = bd[x,y]
                if po>0:
                    for d in dire:
                        step=0
                        xn,yn = x,y
                        for s in range(3):
                            # 计算下一个点的位置
                            xn,yn = xn+d[0],yn+d[1]
                            # 判断点是否在区域内
                            if 0<=xn<=self.num_row-1 and 0<=yn<=self.num_col-1:
                                if bd[xn,yn]==po:
                                    step += 1
                                else:
                                    break
                            else:
                                break
                        # 判断走了多少步
                        if step==3:
                            return True
        
        return ret
    
    def changeplayer(self):
        self.player = self.player-1 if self.player == 2 else 2

    def showtip(self):
        txt = '玩家：%s     电脑:%s\n' % tuple(self.icon[1:])
        print(txt)
        return txt
        
    def showplayer(self):
        #print('当前轮到: [%s]玩家%d ' % (self.playericon(), self.player) )
        txt = '当前轮到: [%s]%s ' % (self.playericon(), self.players[self.player-1])
        print(txt)
        return txt
    
    def playericon(self):
        return self.icon[self.player]
        
class Column:
    def __init__(self, r):
        '''each column has r rows'''
        self.rows = [0]*r
    
    def drop_disk(self,player):
        if 0 in set(self.rows):
            index = self.rows.index(0)
            self.rows[index] = player
            return True
        else:
            print('下错了，该列已经放不下棋子')
            return False
    
    def getdata(self):
        return self.rows

    def setdata(self,dat):
        self.rows = dat
        


# 棋串得分计算公式
def chessvalue(l,d1,d2):
    
    if l<4 and  d1==d2==0: return 0
    value = 10**(l+2) 
    k1 = d1*d2
    k2 = d1+d2
    
    v1 = 0 if d1==0 else 10**(l+1)/d1**3
    v2 = 0 if d2==0 else 10**(l+1)/d2**3
    value += v1+v2
    if k1>0:
        value += (10**(l+1) / k1**1.5)
    if k2>0:
        value += (10**(l) / k2)
    #value *= 100
    return int(value)


# 人工智能类
    
class AI:
    def __init__(self,bd):
        # 复制状态，注意是已经旋转过的
        self.board = bd.status().copy()
        pass
    
    def score(self):
        allchess = self.all_chess()
        v = list(map(lambda y: sum(map(lambda x:x[-1], y)), allchess))
        return v
        
    def nextstep(self, player):
        '''预测出下一步最好的棋应该下在哪里'''
        
        '''计算下一步可下棋的位置'''
        pos = np.where (self.board[0]==0)[0]+1
        #print('pos:', pos)
        
        #模拟在某一列下一个棋
        def dropstep(bd, col, player):
            bd = bd.copy()
            #print('-'*30)
            #print(bd)
            k = bd[:, col][::-1]
            if 0 in k.tolist():
                row = bd.shape[0] - k.tolist().index(0) - 1
                bd[row,col] = player
            return bd
        
        # 当前得分
        current_score = np.array(self.score())
        #print('current_score:', current_score)
        
        #分别计算每一个可下位置下棋后的状态得分变化值
        def nextscore(bd, current_score, player):
            nplayer = 1 if player==2 else 2
            f_score = np.array([current_score] * bd.shape[1])
            '''计算下一步可下棋的位置'''
            pos = np.where (self.board[0]==0)[0]
            for p in pos:
                #模拟下一步棋后的状态
                nbd = dropstep(bd, p, player)
                # 计算状态及得分
                v_chess = self.all_chess(board=nbd)
                v_score = np.array(list(map(lambda y: sum(map(lambda x:x[-1], y)), v_chess)))

                '''
                # 求另一个对手在同一列下棋后的得分
                nnbd = dropstep(nbd, p, nplayer)
                nn_chess = self.all_chess(board=nnbd)
                nn_score = np.array(list(map(lambda y: sum(map(lambda x:x[-1], y)), nn_chess)))
                '''
                # 进阶搜索
                # 求另一个对手的所有下棋后的, 对方得分最高的分值
                nn_score = np.array([0,0])
                npos = np.where(nbd[0]==0)[0]
                if npos.shape[0]>0:
                    nn_s = []
                    for npo in npos:
                        nnbd = dropstep(nbd, npo, nplayer)
                        nn_chess = self.all_chess(board=nnbd)
                        nn_score1 = list(map(lambda y: sum(map(lambda x:x[-1], y)), nn_chess))
                        # 如果想加深 还可以在这里再加一层深度


                        nn_s.append(nn_score1)

                    # 得到最后的局面
                    nn_score = np.array(nn_s).max(axis=0)   #.sum()

                #nn_score = nextscore(nbd, v_score, nplayer)
                #print('nn_score:', nn_score)
                #print('v_score:', v_score)
                # 得分思路：自己下完后的局面，加上对方下完后的最好局面。
                pp = [1,0] if player==1 else [0, 1]
                v_score = v_score*pp + nn_score * (pp[::-1])

                #npp = [1,-2] if nplayer==2 else [-2, 1]
                #v_score = v_score * pp + (nn_score * npp * 0.8)
                #v_score = v_score + nn_score * 1.8

                f_score[p] = v_score.tolist()
            f_score -= current_score
            return f_score

        #print('f_score:' , f_score)
        # 计算差值
        #score_change = f_score - current_score
        
        score_change = nextscore(self.board, current_score, player )

        #print('score_change:', score_change)
        # 得分变化计算方式：已方增加值  对方降低值
        pp = [1,-1] if player==1 else [-1, 1] 
        score_change = score_change * pp
        t_score = score_change.sum(axis=1) # 

        #t_score = ((score_change[:,0]**2 + score_change[:,1]**2)**0.5).astype(int)


        #print(t_score)
        # 排序
        idx = np.array(t_score).argsort()[::-1] + 1
        #print('idx:', idx)
        # 那些不能下棋的位置，就不要返回了
        idx = list(filter(lambda x: x in pos , idx))
        #print('sorted:', idx)
        nt = np.sort(t_score,axis=0)[::-1]
        #ns = np.sort(score_change,axis=0)[::-1]
        #nf =  np.sort(f_score,axis=0)[::-1]
        
        rpos = list(zip(idx, nt.tolist())) #, ns.tolist(), nm.tolist()
        return rpos


    
    def all_chess(self, board=None):
        '''盘面分析，获取所有的棋串，并计算得分'''
        
        #判断子串重复
        check_repeat = lambda m,n: all( map(lambda x:x in n, m))
        
        ret = [[],[]]
        # 每个点沿右，右下，下，左下4个方向最多走3步，每一步都必须与当前点同色，如果能走到3步则表示获胜
        dire = [[0,1],[1,1],[1,0],[1,-1]]
        if board is None:
            bd = self.board
        else:
            bd = board
        for x in range(bd.shape[0]):
            for y in range(bd.shape[1]):
                po = bd[x,y]
                if po>0:
                    #ret[po-1].append([(x,y)])
                    for d in dire:
                        step=0
                        chess = []
                        #chess.append(po)
                        chess.append((x,y))
                        xn,yn = x,y
                        #----- 计算气度1:d1
                        d1 = 0
                        px, py = x-d[0],y-d[1]
                        if 0<=px<bd.shape[0] and 0<=py<bd.shape[1]:
                            if bd[px,py]==0:
                                k = bd[:, py][::-1][:bd.shape[0]-px]
                                d1 = len(list(filter(lambda x:x==0,k)))
                        #-----
                        #最大深入3级
                        for s in range(3):
                            # 计算下一个点的位置
                            xn,yn = xn+d[0],yn+d[1]
                            # 判断点是否在区域内
                            if 0<=xn<bd.shape[0] and 0<=yn<bd.shape[1]:
                                if bd[xn,yn]==po:
                                    chess.append((xn,yn))
                                    step += 1
                                else:
                                    break
                            else:
                                break
                        #----- 计算气度2：d2
                        d2 = 0
                        #xn,yn = xn+d[0],yn+d[1]
                        if 0<=xn<bd.shape[0] and 0<=yn<bd.shape[1]:
                                if bd[xn,yn]==0:
                                    k = bd[:, yn][::-1][:bd.shape[0]-xn]
                                    d2 = len(list(filter(lambda x:x==0,k)))
                        #-----
                        # 记录棋串
                        #if step>0:
                        # 判断是重重复
                        if not any(map(lambda x:check_repeat(chess,x[0]), ret[po-1])):
                            value = chessvalue(len(chess),d1,d2)
                            ret[po-1].append([chess,[len(chess), d1, d2],value])
                        # 记录棋串长度，气度1，气度2
        # 棋串排序,按棋长
        #mysort = lambda r: sorted(r,key=lambda x:-len(x[0]))
        # 棋串排序,按棋串得分
        mysort = lambda r: sorted(r,key=lambda x:-x[-1])
        ret = list(map(mysort,ret))
        return ret
    

def AIrobot ():
    print()
    N=4
    board = Board(N)
    max_step = (N+2)*(N+3)
    win = False
    step = 0
    # 记录步骤
    allstep, nstep = [], []
    while not win and step<max_step:
        board.display()
        if board.winner: 
            if nstep: allstep.append(nstep)
            print('\n胜利！ 获胜者：%s%s' % (board.icon[board.winner], board.players[board.winner-1]) )
            break    
        # ----- 加入AI -----
        ai = AI(board)
        v1, v2 = ai.score()
        print('当前得分评估  玩家:{0:,d}  电脑:{1:,d}'.format(v1,v2))
        recommand = ai.nextstep(board.player)
        print('AI推荐走法：',recommand )
        if board.player ==2:
            print("电脑走棋：%d" % recommand[0][0])
            board.drop_disk(recommand[0][0])
            step += 1
            # 记录
            nstep.append(recommand[0][0])
            allstep.append(nstep)
            nstep = []
            
            board.display()

            ai = AI(board)
            v1, v2 = ai.score()
            print('当前得分评估  玩家:{0:,d}  电脑:{1:,d}'.format(v1,v2))

        print()
        # ----- AI结束 -----
        if board.winner: 
            if nstep: allstep.append(nstep)
            print('\n胜利！ 获胜者：%s%s' % (board.icon[board.winner], board.players[board.winner-1]) )
            break
        q = 0
        while 1:
            try:
                c = input('请输入位置(1-%d):' % (N+2)).strip()
                if c=='q':
                    q = 1
                    break
                c = int(c)
            except :
                continue
                pass

            ret = board.drop_disk(c)
            if ret:
                nstep.append(c)
                step += 1
                break

        
        if q:break


    print('走棋结束!')
    print('对弈过程:%s' % str(allstep)) 

# -----------------------------------------
import argparse
import os
import sys
import re
import logging

from flask import Flask, request, render_template, jsonify, abort, make_response
from flask import url_for, Response, json, session, send_from_directory

# 版本号
gblVersion = '1.0.3'


def txtprocess (txtinfo):
    txtarr = txtinfo.split('\n')
    txtout = ""
    for i,x in enumerate(txtarr):
        if i>8:
            txtout += "<div class=\"line\">" + x + "</div>\n"
        else:
            tmp = ""
            for i,n in enumerate(x):
                tmp += "<div class=\"sp\">" + n + "</div>"
            txtout += "<div class=\"line\">" + tmp + "</div>\n"
    return txtout
    
# Flask 服务端
def HttpServer (args):

    # 参数处理
    ip = args.ip
    port = args.port

    logging.info( ('重力四子棋 v' + gblVersion ).center(40, '-') )

    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.urandom(24)

    @app.route('/favicon.ico')
    def favicon():
        return app.send_static_file('favicon.ico')

    @app.route('/', methods=['GET'])
    def index():
        print('='*40)
        bd = session.get('bd')
        print('session:' , bd)
        if bd:
            print('恢复棋盘...')
            board = Board(4)
            board.setjson(bd)

        else:
            print('创建新棋盘...')
            board = Board(4)
            #print(board.getjson())
            session['bd'] = board.getjson()

        txtinfo = board.display()
        txtinfo = txtprocess(txtinfo)
        # 加入获胜比分 2020/11/12
        print('当前对战比分:%s' % board.score)
        score = '<span>&nbsp;%d</span>&nbsp;:&nbsp;<span>%d</span>' % tuple(board.score) #.tolist()
        score_text = '你来试一把？'
        if board.score[0]==board.score[1]:
            score_text = '人类和AI打成平手了，快来挑战！'
        if board.score[0]>board.score[1]:
            score_text = '人类领先了，你也来挑战一下！'
        if board.score[0]<board.score[1]:
            score_text = '人类落后了，快点来挑战吧！'

        return render_template('index.html', version=gblVersion, txtinfo=txtinfo, score=score,score_text=score_text )
    
    
    @app.route('/nextstep', methods=['POST'])
    def nextstep():
        res = {}

        bd = session.get('bd')
        print('session:' , bd)
        if bd:
            print('恢复棋盘...')
            board = Board(4)
            board.setjson(bd)
            #txtinfo = board.display()

        # 开始处理
        col = request.values.get('col')
        print('玩家提交:',col)
        if col=='n':
            #session['bd'] = None
            print('创建新棋盘...')
            board = Board(4)
            print(board.getjson())
            txtinfo = board.display()
            session['bd'] = board.getjson()
            res['status'] = 'OK'
            res['txtinfo'] = txtinfo
            return jsonify(res)

        # 结束状态下只能重新开局
        if board.winner!=0:
            res['status'] = 'Err'
            res['code'] = 1
            return jsonify(res)

        if col in '123456':
            col = int(col)
            print('玩家下棋:',col)
            ret = board.drop_disk(col)
            print(ret)
            if ret:
                if board.winner==1: 
                    txtinfo = board.display()
                    txtinfo += '\n你胜利了！ 太棒了，你战胜了AI！'
                    #'\n胜利！ 获胜者：%s%s' % (board.icon[board.winner], board.players[board.winner-1])
                    #session['bd'] = None
                else:
                    ai = AI(board)
                    v1, v2 = ai.score()
                    print('当前得分评估  玩家:{0:,d}  电脑:{1:,d}'.format(v1,v2))
                    recommand = ai.nextstep(board.player)
                    print('AI推荐走法：',recommand )
                    if board.player ==2:
                        print("电脑走棋：%d" % recommand[0][0])
                        board.drop_disk(recommand[0][0])

                        txtinfo = board.display()
                        txtinfo += "\n电脑走棋：%d" % recommand[0][0]
                        
                        '''
                        ai = AI(board)
                        v1, v2 = ai.score()
                        print('当前得分评估  玩家:{0:,d}  电脑:{1:,d}'.format(v1,v2))
                        '''

                        if board.winner==2: 
                            txtinfo += '\nAI获胜！ 别灰心，再加油噢!'
                        else:
                            if board.winner==3:
                                txtinfo += '\n双方战平！再来一局吧！'
                # 保存状态
                session['bd'] = board.getjson()
                res['status'] = 'OK'
                res['txtinfo'] = txtinfo
            else:
                res['status'] = 'Err'
                res['code'] = 2

        print('res:%s' % str(res))
        return jsonify(res)


    logging.info('正在启动服务，请稍候...')
    app.run(
        host = args.ip,
        port = args.port,
        debug = True 
    )


if __name__ == '__main__':
    pass
    #################################################################################################
    # 指定日志
    logging.basicConfig(level = logging.DEBUG,
                format='[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename= os.path.join('./', 'server.log'),
                filemode='a'
                )
    #################################################################################################
    # 定义一个StreamHandler，将 INFO 级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(filename)s [line:%(lineno)d] %(levelname)s %(message)s')
    #formatter = logging.Formatter('[%(asctime)s]%(levelname)s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    #################################################################################################

    parser = argparse.ArgumentParser(description='重力四子棋WEB服务端')
    parser.add_argument('--ip', type=str, default="0.0.0.0", help='IP地址')
    parser.add_argument('--port', type=int, default=8100, help='端口号')

    args = parser.parse_args()


    HttpServer(args)