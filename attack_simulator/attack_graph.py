import random
import logging
from typing import Dict
import numpy as np

class AttackStep:

    def __init__(self, name='', step_type='or', ttc=1, reward=0, children={}, true_positive=1.0, false_positive=0.0, deterministic=False):
        self.name = name
        self.step_type = step_type
        if deterministic:
            self.ttc = ttc
            self.reward = reward
        else:
            self.ttc = int(np.random.exponential(scale=ttc))
            self.reward = int(np.random.exponential(scale=reward))
        self.children = children
        self.parents = set()
        self.true_positive = true_positive
        self.false_positive = false_positive
        self.enabled = True


class AttackGraph:

    def __init__(self, deterministic=False, early_flag_reward=1000, late_flag_reward=10000, graph_size='large'):
        self.deterministic = deterministic
        self.early_flag_reward = early_flag_reward
        self.late_flag_reward = late_flag_reward
        self.graph_size = graph_size
        self.reset()

    def reset(self):
        logger = logging.getLogger("simulator")
        # These are the services and hosts that the defender is at liberty to disable in order to protect the computer network.
        self.enabled_services = dict()
        # Disabling a host, e.g. lazarus, will also disable all of its services
        self.enabled_services['lazarus'] = True
        # The defender can limit disablement to a single service, such as the ftp server on lazarus
        self.enabled_services['lazarus.ftp'] = True
        self.enabled_services['lazarus.tomcat'] = True
        self.enabled_services['energetic_bear'] = True
        self.enabled_services['energetic_bear.apache'] = True
        self.enabled_services['sea_turtle'] = True
        self.enabled_services['sea_turtle.telnet'] = True
        self.enabled_services['buckeye'] = True
        self.enabled_services['buckeye.firefox'] = True
        self.enabled_services['cloud_hopper'] = True
        self.enabled_services['cloud_hopper.smb'] = True
        self.enabled_services['cloud_function'] = True
        self.enabled_services['fancy_bear.ssh'] = True
        self.enabled_services['fancy_bear'] = True
        self.enabled_services['wifi_host.http_server'] = True
        self.enabled_services['wifi_host'] = True

        logger.info(str(len(self.enabled_services)) +
                    " possible defender actions.")

        self.attack_steps: Dict[str, AttackStep] = {}

        # Here the attack logic is defined. The below is a model of the EN2720 course.
        if self.graph_size == 'large':
                self.attack_steps['wifi_host.http_server.flag_18dd8f.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['wifi_host.http_server.gather_information'] = AttackStep(ttc=3, children={'wifi_host.http_server.flag_18dd8f.capture'}, deterministic=self.deterministic)
                self.attack_steps['wifi_host.http_server.connect'] = AttackStep(children={'wifi_host.http_server.gather_information'}, deterministic=self.deterministic)

                self.attack_steps['wifi_network.map'] = AttackStep(ttc=10, children={'wifi_host.http_server.connect'}, deterministic=self.deterministic)
                self.attack_steps['wifi_network.connect'] = AttackStep(step_type='and', children={'wifi_network.map'}, deterministic=self.deterministic)
                self.attack_steps['wifi_network.flag_d582aa.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['wifi_network.decrypt_traffic'] = AttackStep(step_type='and', ttc=5, children={'wifi_network.flag_d582aa.capture'}, deterministic=self.deterministic)
                self.attack_steps['wifi_network.obtain_credentials'] = AttackStep(ttc=10, children={
                                                                                  'wifi_network.connect', 'wifi_network.decrypt_traffic'}, deterministic=self.deterministic)
                self.attack_steps['wifi_network.find_credentials'] = AttackStep(children={'wifi_network.obtain_credentials'}, deterministic=self.deterministic)
                self.attack_steps['wifi_network.capture_traffic'] = AttackStep(ttc=5, children={
                                                                               'wifi_network.connect', 'wifi_network.find_credentials', 'wifi_network.decrypt_traffic'}, deterministic=self.deterministic)

                self.attack_steps['fancy_bear.gather_information'] = AttackStep(ttc=5, children={'wifi_network.capture_traffic'}, deterministic=self.deterministic)

                self.attack_steps['fancy_bear.ssh.login'] = AttackStep(step_type='and', children={
                                                                       'fancy_bear.gather_information'}, deterministic=self.deterministic)
                self.attack_steps['fancy_bear.ssh.obtain_credentials'] = AttackStep(children={'fancy_bear.ssh.login'}, deterministic=self.deterministic)
                self.attack_steps['fancy_bear.ssh.connect'] = AttackStep(children={'fancy_bear.ssh.login'}, deterministic=self.deterministic)

                self.attack_steps['cloud_function.flag_831865.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['cloud_function.flag_d8d9da.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['cloud_function.exploit_vulnerability'] = AttackStep(children={'cloud_function.flag_831865.capture'}, deterministic=self.deterministic)
                self.attack_steps['cloud_function.find_vulnerability'] = AttackStep(children={'cloud_function.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['cloud_function.gather_information'] = AttackStep(children={'cloud_function.find_vulnerability', 'cloud_function.flag_d8d9da.capture'}, deterministic=self.deterministic)

                self.attack_steps['cloud_bucket.flag_21077e.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['cloud_bucket.find_credentials'] = AttackStep(children={'fancy_bear.ssh.obtain_credentials'}, deterministic=self.deterministic)
                self.attack_steps['cloud_bucket.list'] = AttackStep(children={'cloud_function.gather_information', 'cloud_bucket.find_credentials', 'cloud_bucket.flag_21077e.capture'}, deterministic=self.deterministic)

                self.attack_steps['cloud_hopper.gather_information'] = AttackStep(ttc=5, children={'cloud_bucket.list'}, deterministic=self.deterministic)
                self.attack_steps['cloud_hopper.flag_93b00a.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['cloud_hopper.terminal_access'] = AttackStep(children={'cloud_hopper.gather_information', 'cloud_hopper.flag_93b00a.capture'}, deterministic=self.deterministic)
                self.attack_steps['cloud_hopper.smb.exploit_vulnerability'] = AttackStep(ttc=10, children={'cloud_hopper.terminal_access'}, deterministic=self.deterministic)
                self.attack_steps['cloud_hopper.smb.find_vulnerability'] = AttackStep(ttc=5, children={'cloud_hopper.smb.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['cloud_hopper.smb.connect'] = AttackStep(children={'cloud_hopper.smb.find_vulnerability'}, deterministic=self.deterministic)

                self.attack_steps['hidden_network.map'] = AttackStep(ttc=10, children={
                                                                     'cloud_hopper.smb.connect', 'fancy_bear.ssh.connect'}, deterministic=self.deterministic)
                self.attack_steps['hidden_network.connect'] = AttackStep(children={'hidden_network.map'}, deterministic=self.deterministic)

                self.attack_steps['buckeye.flag_2362e5.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['buckeye.flag_5d402e.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['buckeye.firefox.flag_14ce18.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['buckeye.escalate_to_root'] = AttackStep(children={'hidden_network.connect', 'buckeye.flag_2362e5.capture'}, deterministic=self.deterministic)
                self.attack_steps['buckeye.exploit_vulnerability'] = AttackStep(ttc=10, children={'buckeye.escalate_to_root'}, deterministic=self.deterministic)
                self.attack_steps['buckeye.find_vulnerability'] = AttackStep(ttc=5, children={'buckeye.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['buckeye.terminal_access'] = AttackStep(children={'buckeye.find_vulnerability', 'buckeye.flag_5d402e.capture'}, deterministic=self.deterministic)
                self.attack_steps['buckeye.firefox.exploit_vulnerability'] = AttackStep(ttc=10, children={'buckeye.terminal_access'}, deterministic=self.deterministic)
                self.attack_steps['buckeye.firefox.find_vulnerability'] = AttackStep(ttc=5, children={'buckeye.firefox.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['buckeye.firefox.connect'] = AttackStep(children={'buckeye.firefox.find_vulnerability'}, deterministic=self.deterministic)

                self.attack_steps['sea_turle.flag_6be6ef.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['sea_turle.flag_f9038f.capture'] = AttackStep(reward=self.late_flag_reward, deterministic=self.deterministic)
                self.attack_steps['sea_turtle.capture_traffic'] = AttackStep(ttc=10, children={
                                                                             'buckeye.firefox.connect', 'buckeye.firefox.flag_14ce18.capture'}, deterministic=self.deterministic)
                self.attack_steps['sea_turtle.escalate_to_root'] = AttackStep(ttc=50, children={'sea_turtle.capture_traffic', 'sea_turle.flag_6be6ef.capture'}, deterministic=self.deterministic)
                self.attack_steps['sea_turtle.exploit_vulnerability'] = AttackStep(ttc=50, children={'sea_turtle.escalate_to_root'}, deterministic=self.deterministic)
                self.attack_steps['sea_turtle.find_vulnerability'] = AttackStep(ttc=50, children={'sea_turtle.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['sea_turtle.telnet.login'] = AttackStep(step_type='and', children={
                                                                          'sea_turtle.find_vulnerability', 'sea_turle.flag_f9038f.capture'}, deterministic=self.deterministic)
                self.attack_steps['sea_turtle.telnet.obtain_credentials'] = AttackStep(children={'sea_turtle.telnet.login'}, deterministic=self.deterministic)
                self.attack_steps['sea_turtle.telnet.connect'] = AttackStep(children={'sea_turtle.telnet.login'}, deterministic=self.deterministic)

        if self.graph_size == 'medium' or self.graph_size == 'large':
                self.attack_steps['energetic_bear.flag_73cb43.capture'] = AttackStep(reward=self.early_flag_reward, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.flag_3b2000.capture'] = AttackStep(reward=self.early_flag_reward, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.flag_de3b1c.capture'] = AttackStep(reward=self.early_flag_reward, deterministic=self.deterministic)
                if self.graph_size == 'large':
                        self.attack_steps['energetic_bear.capture_traffic'] = AttackStep(ttc=5, children={
                                                                                 'buckeye.firefox.connect', 'buckeye.firefox.flag_14ce18.capture'}, deterministic=self.deterministic)
                else:
                        self.attack_steps['energetic_bear.capture_traffic'] = AttackStep(ttc=5, children={}, deterministic=self.deterministic)

                self.attack_steps['energetic_bear.escalate_to_root'] = AttackStep(ttc=20, children={
                                                                                  'energetic_bear.capture_traffic', 'energetic_bear.flag_73cb43.capture'}, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.exploit_vulnerability'] = AttackStep(ttc=50, children={'energetic_bear.escalate_to_root'}, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.find_vulnerability'] = AttackStep(ttc=50, children={'energetic_bear.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.terminal_access'] = AttackStep(ttc=10, children={
                                                                                 'energetic_bear.find_vulnerability', 'energetic_bear.flag_3b2000.capture'}, deterministic=self.deterministic)

                self.attack_steps['energetic_bear.apache.flag_521bce.capture'] = AttackStep(reward=self.early_flag_reward, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.apache.exploit_vulnerability'] = AttackStep(ttc=20, children={
                                                                                              'energetic_bear.terminal_access', 'energetic_bear.flag_de3b1c.capture'}, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.apache.find_vulnerability'] = AttackStep(ttc=30, children={'energetic_bear.apache.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.apache.gather_information'] = AttackStep(ttc=10, children={
                                                                                           'energetic_bear.apache.find_vulnerability', 'energetic_bear.apache.flag_521bce.capture'}, deterministic=self.deterministic)
                self.attack_steps['energetic_bear.apache.connect'] = AttackStep(children={'energetic_bear.apache.gather_information'}, deterministic=self.deterministic)

                self.attack_steps['lazarus.flag_cd699a.capture'] = AttackStep(reward=self.early_flag_reward, deterministic=self.deterministic)
                if self.graph_size == 'large':
                        self.attack_steps['lazarus.find_credentials'] = AttackStep(ttc=10, children={'sea_turtle.telnet.obtain_credentials'}, deterministic=self.deterministic)
                else:
                        self.attack_steps['lazarus.find_credentials'] = AttackStep(ttc=10, deterministic=self.deterministic)
                self.attack_steps['lazarus.terminal_access'] = AttackStep(children={'lazarus.find_credentials', 'lazarus.flag_cd699a.capture', 'lazarus.ftp.flag_adcb1f.capture'}, deterministic=self.deterministic)

                self.attack_steps['lazarus.tomcat.flag_90b353.capture'] = AttackStep(reward=self.early_flag_reward, deterministic=self.deterministic)
                self.attack_steps['lazarus.tomcat.exploit_vulnerability'] = AttackStep(ttc=10, children={'lazarus.terminal_access'}, deterministic=self.deterministic)
                self.attack_steps['lazarus.tomcat.find_vulnerability'] = AttackStep(ttc=10, children={'lazarus.tomcat.exploit_vulnerability'}, deterministic=self.deterministic)
                self.attack_steps['lazarus.tomcat.dictionary_attack'] = AttackStep(ttc=10, children={
                                                                                   'lazarus.tomcat.find_vulnerability', 'lazarus.tomcat.flag_90b353.capture'}, deterministic=self.deterministic)
                self.attack_steps['lazarus.tomcat.gather_information'] = AttackStep(ttc=5, children={'lazarus.tomcat.dictionary_attack'}, deterministic=self.deterministic)
                self.attack_steps['lazarus.tomcat.connect'] = AttackStep(children={'lazarus.tomcat.gather_information'}, deterministic=self.deterministic)

        self.attack_steps['lazarus.ftp.flag_adcb1f.capture'] = AttackStep(reward=self.early_flag_reward, deterministic=self.deterministic)
        self.attack_steps['lazarus.ftp.login'] = AttackStep(children={'lazarus.ftp.flag_adcb1f.capture'}, deterministic=self.deterministic)
        self.attack_steps['lazarus.ftp.dictionary_attack'] = AttackStep(ttc=100, children={'lazarus.ftp.login'}, deterministic=self.deterministic)
        self.attack_steps['lazarus.ftp.connect'] = AttackStep(children={'lazarus.ftp.dictionary_attack'}, deterministic=self.deterministic)

        if self.graph_size == 'large':
                self.attack_steps['office_network.map'] = AttackStep(ttc=10, children={
                                                             'lazarus.ftp.connect', 'energetic_bear.apache.connect', 'lazarus.tomcat.connect', 'sea_turtle.telnet.connect'}, deterministic=self.deterministic)
        if self.graph_size == 'medium':
                self.attack_steps['office_network.map'] = AttackStep(ttc=10, children={
                                                             'lazarus.ftp.connect', 'energetic_bear.apache.connect', 'lazarus.tomcat.connect'}, deterministic=self.deterministic)
        if self.graph_size == 'small':
                self.attack_steps['office_network.map'] = AttackStep(ttc=10, children={
                                                             'lazarus.ftp.connect'}, deterministic=self.deterministic)

        self.attack_steps['office_network.connect'] = AttackStep(children={'office_network.map'}, deterministic=self.deterministic)
        self.attack_steps['internet.connect'] = AttackStep(children={'office_network.connect'}, deterministic=self.deterministic)

        self.record_parents()
        self.size = len(self.attack_steps)

        logger.info(str(self.size) + " attack steps.")

    def record_parents(self):
        # And steps need to know which their parents are.
        for parent in self.attack_steps:
            for child in self.attack_steps[parent].children:
                self.attack_steps[child].parents.add(parent)

    def disable(self, service):
        # Disable service
        self.enabled_services[service] = False
        # Disconnect all attack steps that match the service.
        for step_name in self.attack_steps:
            if self.attack_steps[step_name].enabled and service in step_name:
                self.attack_steps[step_name].enabled = False
        # Also disable subservices (e.g. fancy_bear.ssh should be disabled when fancy_bear is)
        for subservice in self.enabled_services:
            if self.enabled_services[subservice] and service in subservice and service != subservice:
                self.enabled_services[subservice] = False

    def generate_graphviz_file(self):
        f = open("graphviz.dot", "w")
        f.write("digraph G {\n")

        for parent in self.attack_steps:
            for child in self.attack_steps[parent].children:
                f.write(f"\"{parent}\" -> \"{child}\";\n") 
        f.write("}\n")
        f.close()
        print("Generated a GraphViz file of the attack graph which, e.g., can be viewed at https://dreampuf.github.io/GraphvizOnline.")
                
