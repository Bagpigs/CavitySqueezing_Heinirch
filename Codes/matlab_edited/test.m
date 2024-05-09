%Pair Productions with Different Classical Seeds (Truncated Wigner Simulations)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DEFINE PARAMETERS OF SIMULATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
clc


%Constants
hbar = 1.054571628*(10^-34);
mRb87 = 1.443160648*(10^-25);
epsilon0= 8.8541878128e-12;
lam_M=790.02e-9;
E0=403; %V/m electric field per cavity photon;
c=2.9979e8;
omegaRec = hbar*2*pi*pi./(mRb87.*lam_M.*lam_M)
omegaR=omegaRec/1000;
p=0.7024*1e6;% First order splitting of Rb87 in Hz
q=144; % 2md second order splitting of Rb87 in Hz

%PARAMETERS
N=80000;% Atom Number
DeltaN=0.0*N; % Fluctuations of Atom Number
tbounds=[0 0.2]; % Evolution Time in ms
eta=2*pi*1.7e3; % Raman Coupling

Npoints   = 2000;
time=linspace(tbounds(1),tbounds(2),Npoints);
Kappa=2*pi*1.25e6; % Cavity Losses in Hz
omegaZ=2*pi*7.09*1e6; % Zeemansplitting in Hz
deltaC=-2*pi*25.8e6;% Cavity Detuning in Hz




%Two Photon Detunings
delta_p=(deltaC+omegaZ); %Two Photon Detunings for + channel
delta_m=(deltaC-omegaZ); %Two Photon Detunings for - channel
omega0=0.5*(4*omegaRec+2*pi*q*(omegaZ/2/pi/p).^2)/1000;



% Caculate Two Photon And Four Photon Couplings
x_p=(eta^2*delta_p./(delta_p.^2+(Kappa)^2))/1000;% Pair coupling for chi_+ Channel
Gamma_p=(eta^2*Kappa./(delta_p.^2+(Kappa)^2))/1000;% Dissipative coupling for chi_+ Channel
x_m=(eta^2*delta_m./(delta_m.^2+(Kappa)^2))/1000;% Pair coupling for chi_- Channel
Gamma_m=(eta^2*Kappa./(delta_m.^2+(Kappa)^2))/1000;% Pair coupling for chi_+ Channel



%Define Ramps
ramptype='rampup_sshape_hold';
x_p_bounds=[1 1]; % [1 1]: Ramp Not active, [0 1]: Ramp Active
gamma_p_bounds=[1 1];
x_m_bounds=[1 1];
gamma_m_bounds=[1 1];
x_p_bounds=x_p_bounds*x_p;
gamma_p_bounds=gamma_p_bounds*Gamma_p;
x_m_bounds=x_m_bounds*x_m;
gamma_m_bounds=gamma_m_bounds*Gamma_m;


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DO SMULATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_list=[0]; %!!! List of average initial pair number from classical seeds
Nrealiz=1000; %!!! Number of TW simulations pro setting (used for averaging)
scalecoupling_k2=1; % Scale Coupling to m=0, k_x=+-2k mode, set to 0 to supress coupling to this mode and 1 to consider it correctily
SeedType='Deterministic';

cmap=jet(length(n_list));
rhoM1_vec=zeros(Npoints,Nrealiz,length(n_list));
rho1_vec=zeros(Npoints,Nrealiz,length(n_list));
rhoM1_M_vec=zeros(Npoints,Nrealiz,length(n_list));
rho1_M_vec=zeros(Npoints,Nrealiz,length(n_list));
rho0_vec=zeros(Npoints,Nrealiz,length(n_list));


phiM1_vec=zeros(Npoints,Nrealiz,length(n_list));
phi1_M_vec=zeros(Npoints,Nrealiz,length(n_list));

rho5_vec =zeros(Npoints,Nrealiz,length(n_list));


for nn=1:length(n_list)
    % Choose average pair occupation
    nSeed=n_list(nn);
    for jj=1:Nrealiz
    N_temp=normrnd(N,DeltaN); % Atom Number for Realization


    switch SeedType
        case 'Deterministic'
            %Deterministic Seed
            N_temp_therm=nSeed;
            N_temp_therm2=nSeed;

        case 'Poisson'
            N_temp_therm=poissrnd(nSeed);
            N_temp_therm2=poissrnd(nSeed);

    end
        %Classical Occupation
        phi_initial=zeros(1,6);
        phi_initial(1,1) = sqrt(N_temp); % All atoms in mF=0
        phi_initial(1,2) = sqrt(N_temp_therm); % Atoms in mF=1,+k
        phi_initial(1,3) = sqrt(N_temp_therm2); % Atoms in mF=-1,-k

        phi_initial(1,4) =  sqrt(N_temp_therm); % Atoms in mF=-1,+k
        phi_initial(1,5) =  sqrt(N_temp_therm2); % Atoms in mF=1,-k
        phi_initial(1,6) =  0; % Atoms in mF=0,+-2k_x

        %Sample Quantum 1/2 noise
        phi_initial(1,1)=phi_initial(1,1)+ sqrt(0.5)*normrnd(0,1)+ 1i*sqrt(0.5)*normrnd(0,1);
        phi_initial(1,2)=phi_initial(1,2)+ sqrt(0.5)*normrnd(0,1)+ 1i*sqrt(0.5)*normrnd(0,1);
        phi_initial(1,3)=phi_initial(1,3)+ sqrt(0.5)*normrnd(0,1)+ 1i*sqrt(0.5)*normrnd(0,1);
        phi_initial(1,4)=phi_initial(1,4)+ sqrt(0.5)*normrnd(0,1)+ 1i*sqrt(0.5)*normrnd(0,1);
        phi_initial(1,5)=phi_initial(1,5)+ sqrt(0.5)*normrnd(0,1)+ 1i*sqrt(0.5)*normrnd(0,1);
        phi_initial(1,6)=phi_initial(1,6)+ sqrt(0.5)*normrnd(0,1)+ 1i*sqrt(0.5)*normrnd(0,1);
        %Normailize
        %phi_initial=phi_initial*N_temp/sum(abs(phi_initial(1,:)).^2);


        % SOLVE AND PLOT EOMS
        tic;
            options = odeset('RelTol',1e-8,'AbsTol',1e-8);
            [t,phiout]   = ode45(@(tdummy,phidummy) eomsSixAtomicModes_PairProduction_NOCAVITY(tdummy,phidummy,tbounds,x_p_bounds,gamma_p_bounds,x_m_bounds,gamma_m_bounds,1,omega0,omegaR,scalecoupling_k2,ramptype),[tbounds(1) tbounds(2)],phi_initial,options);

            phiTW=interp1(t,phiout,time);
        elapsedTime=toc;
        if mod(jj,50)==1
            disp(['NpSeed=',num2str(n_list(nn)),', EOMs ',num2str(jj), 'out of' ,num2str(Nrealiz),' , TimeForSingleEOM=',num2str(elapsedTime),'s'])
        end

        %Store Results
        rho0_vec(:,jj,nn)=abs(phiTW(:,1)).^2;
        rho1_vec(:,jj,nn)=abs(phiTW(:,2)).^2;
        rhoM1_vec(:,jj,nn)=abs(phiTW(:,3)).^2;
        rhoM1_M_vec(:,jj,nn)=abs(phiTW(:,4)).^2;
        rho1_M_vec(:,jj,nn)=abs(phiTW(:,5)).^2;

        phiM1_vec(:,jj,nn)=phiTW(:,3);
        phi1_M_vec(:,jj,nn)=phiTW(:,5);

        rho5_vec(:,jj,nn)=abs(phiTW(:,6)).^2;
    end

    end



% Calculate Expectation Values and Stds of different Modes
rho1_mean=squeeze(mean(rho1_vec,2));
rho1_std=sqrt(squeeze(var(rho1_vec,0,2)));
rhoM1_mean=squeeze(mean(rhoM1_vec,2));
rhoM1_std=sqrt(squeeze(var(rhoM1_vec,0,2)));

rho1_M_mean=squeeze(mean(rho1_M_vec,2));
rho1_M_std=sqrt(squeeze(var(rho1_M_vec,0,2)));
rhoM1_M_mean=squeeze(mean(rhoM1_M_vec,2));
rhoM1_M_std=sqrt(squeeze(var(rhoM1_M_vec,0,2)));

rho0_mean = squeeze(mean(rho0_vec,2));
rho5_mean = squeeze(mean(rho5_vec,2));
%% Plot Time Evolution
indmax=10;
set(groot,'defaultAxesTickLabelInterpreter','latex');

figure(90);clf
Leg={};
for nn=1:length(n_list)
    subplot(1,1,1)
    hold on
    plot(1000*time,rho1_mean(:,nn),'color',cmap(nn,:),'linewidth',2)
    plot(1000*time,rhoM1_mean(:,nn),'color',cmap(nn,:),'LineStyle','--','linewidth',0.3)
    plot(1000*time,rho5_mean(:,nn),'color',cmap(nn,:),'linewidth',1)
    %
    % plot(1000*time,rho0_mean(:,nn),'color',cmap(nn,:),'linewidth',2)
    % plot(1000*time,rhoM1_M_mean(:,nn),'color',cmap(nn,:),'LineStyle','--','linewidth',0.3)
    % plot(1000*time,rho1_M_mean(:,nn),'color',cmap(nn,:),'linewidth',1)

    title([SeedType, ' Seed, N=',num2str(N)])
    %disp(['2*N*chi_p/(2*pi):',num2str(2*N*x_p/2/pi),' kHz'])

    grid on
    box on
    ylabel('$\langle N_\textnormal{p}\rangle $','interpreter','latex')
    xlabel('$t (\mu s)$','interpreter','latex')
    %set(gca, 'YScale', 'log')
    LegStr=['n_{seed}=',num2str(n_list(nn))];
    Leg=[Leg,LegStr];
    set(gca,'FontSize',16)
    %yticks([0.1 1 10 100 1000 10000 100000])



end
subplot(1,1,1)
legend(Leg)



%% Plot Exemplary Pair Histograms
% tplot=0.0; %time to plot in ms
% nbins=20; %Number of Bins in Histogram
% indThermal=1; % index of thermal seedto plot-> n_seed=n_list(indThermal)
%
% [~,ind] = min(abs(time-tplot));
% npp=squeeze(rhoM1_vec(ind,:,indThermal));
% npm=squeeze(rho1_M_vec(ind,:,indThermal));
%
% Psi_p=squeeze(phiM1_vec(ind,:,indThermal));
% Psi_m=squeeze(phi1_M_vec(ind,:,indThermal));
%
%
% figure(55);clf
% subplot(2,2,1)
% h1 = histogram(npp,nbins);
% h1.Normalization = 'probability';
% h1.FaceColor = cmap(indThermal,:);
% %xlim([0,5000])
% xlabel('N_{p,+}')
% ylabel('p(N_p)')
%
%
%
% title(['n_{seed}=',num2str(n_list(indThermal)),', t=',num2str(tplot),'ms'])
% leg1=['<N_{p,+}> =',num2str(round(mean(npp))),', \sigma/<N_p> =', num2str(round(std(npp)/mean(npp),3))];
% legend(leg1)
% set(gca,'FontSize',15)
% set(gcf,'color','w');
%
% grid on
%
% subplot(2,2,2)
% plot(real(Psi_p),imag(Psi_p),'o')
% xlabel('Re(\psi_{+})')
% ylabel('Im(\psi_{+})')
% grid on
%
% subplot(2,2,3)
% h2 = histogram(npm,nbins);
% h2.Normalization = 'probability';
% h2.FaceColor = cmap(indThermal,:);
%
% xlabel('N_{p,-}')
% ylabel('p(N_p)')
% set(gca,'FontSize',15)
% set(gcf,'color','w');
% grid on
%
% subplot(2,2,4)
% plot(real(Psi_m),imag(Psi_m),'o')
% xlabel('Re(\psi_{-})')
% ylabel('Im(\psi_{-})')
% grid on
%
