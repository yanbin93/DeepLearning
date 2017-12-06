#!/usr/bin/env python
#-*- coding:utf-8 -*-
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import os

class Emailer(object):
    def __init__(self,sender='18565589142@163.com',recievers=['1010652868@qq.com']):
        self.username = '18565589142@163.com'
        self.password = 'yb940918'
        self.sender = sender
        self.recievers = ','.join(recievers)
#       self.server = smtplib.SMTP('smtp.163.com',25)

    def send(self,text=None,files=None,subject="logContent"):
        assert text is not None, 'please input email content'
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = self.sender
        msg['To'] = self.recievers
        puretext = MIMEText(text)
        msg.attach(puretext)
        if files is not None:
            for  f in files:
                att = MIMEText(open(f, 'rb').read(), 'base64', 'gb2312')
                att["Content-Type"] = 'application/octet-stream'
                file_name = os.path.basename(f)
                att["Content-Disposition"] = 'attachment; filename=%s' %file_name
                print file_name
                msg.attach(att)
            #filepart = MIMEApplication(open(file_path,'rb').read())
            #filename = file_path.split('/')[-1]
            #print filename
            #filepart.add_header('Content-Disposition','attachment',filename=filename)
            #msg.attach(filepart)
        server = smtplib.SMTP()
        server.connect('smtp.163.com',25)
        server.login(self.username,self.password)
        print "Connection to 163.com"
        server.sendmail(self.sender,self.recievers,msg.as_string())
        server.quit()
        print "Send email sucessfully"

if __name__ == '__main__':
    emailer = Emailer()
    emailer.send(files=['log.txt'],text = 'help')
