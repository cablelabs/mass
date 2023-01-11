FROM ubuntu:18.04
RUN apt-get update


# ns3 requirements

RUN apt-get --fix-missing install

RUN apt-get update

RUN apt-get install -y g++ python3 python3-dev python3-pip pkg-config sqlite3 python3-setuptools git \
    gir1.2-goocanvas-2.0 python-gi python-gi-cairo python-pygraphviz python3-gi python3-gi-cairo \
    python3-pygraphviz gir1.2-gtk-3.0 ipython ipython3 wget gdb

RUN mkdir -p /usr/ns3
WORKDIR /usr
RUN wget http://www.nsnam.org/release/ns-allinone-3.29.tar.bz2
RUN tar -xf ns-allinone-3.29.tar.bz2
RUN cd ns-allinone-3.29 && ./build.py
RUN ln -s /usr/ns-allinone-3.29/ns-3.29/ /usr/ns3/

WORKDIR /sim
ADD mass_basic_example.cc /usr/ns3/ns-3.29/scratch/mass_basic/
ADD mass_wifi_example.cc /usr/ns3/ns-3.29/scratch/mass_wifi/
ADD mass_lte_example.cc /usr/ns3/ns-3.29/scratch/mass_lte/
ADD mass.h /usr/ns3/ns-3.29/scratch/mass_basic/
ADD mass.cc /usr/ns3/ns-3.29/scratch/mass_basic/
ADD mass.h /usr/ns3/ns-3.29/scratch/mass_wifi/
ADD mass.cc /usr/ns3/ns-3.29/scratch/mass_wifi/
ADD mass.h /usr/ns3/ns-3.29/scratch/mass_lte/
ADD mass.cc /usr/ns3/ns-3.29/scratch/mass_lte/
ADD input-defaults.txt /usr/ns3/ns-3.29/
ADD start_container.sh /sim/
CMD ./start_container.sh
