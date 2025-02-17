import subprocess
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import yaml

script_to_run = 'test_eval.py'    # 要运行的Python脚本名称
process = subprocess.Popen(['python3', script_to_run])
process.wait()
print(f"脚本 {script_to_run} 已结束")

with open('util/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)

subject = 'Python脚本运行结束通知'
body = f"您的Python脚本 {config_data['benchmark']}_{config_data['target']}_{script_to_run}已成功运行结束。"
 
msg = MIMEText(body, 'plain', 'utf-8')
msg['Subject'] = Header(subject, 'utf-8')
msg['From'] = '1255450792@qq.com'
msg['To'] = 'yl.xue@outlook.com'
 
smtp_server = 'smtp.qq.com'
smtp_port = 587
sender_email = '1255450792@qq.com'
p = 'ghidczwjidzcggfj'
 
try:
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        print("TLS connection established, server response:", server.ehlo())  # 打印 TLS 连接建立后的响应
        server.login(sender_email, p)
        print("Login successful, server response:", server.ehlo())  # 打印登录后的响应
        server.sendmail(sender_email, [msg['To']], msg.as_string())
        print('邮件发送成功')
except smtplib.SMTPException as e:
    print('邮件发送失败:', str(e))