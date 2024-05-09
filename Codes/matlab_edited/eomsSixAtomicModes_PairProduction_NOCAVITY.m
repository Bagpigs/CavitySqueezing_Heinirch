function dphidt = eomsSixAtomicModes_PairProduction_NOCAVITY(t,phi,tbounds,x_p_bounds,gamma_p_bounds,x_m_bounds,gamma_m_bounds,N,omega0, omegaRec,scalecoupling_k2,ramptype)
%Solve Mean Field Equations of IDJC Hamiltonian: Time and all energy scales in units of sqrt(omega0*omegaC) for


%Ramp TPs with user defined function
chi=N*rampcoupling(t,tbounds,x_p_bounds,ramptype);
gamma=N*rampcoupling(t,tbounds,gamma_p_bounds,ramptype);
chiM=N*rampcoupling(t,tbounds,x_m_bounds,ramptype);
gammaM=N*rampcoupling(t,tbounds,gamma_m_bounds,ramptype);

%Coupling To k=+-2k_x modes in mF=0
a=2/3;
a=(scalecoupling_k2)^2*a;

    dphidt = zeros(6,1);
    dphidt(1) = -1i*chi*(2*conj((phi(1)+sqrt(a)*phi(6)))*phi(2)*phi(3)+ conj(phi(2))*(phi(1)+sqrt(a)*phi(6))*phi(2) + conj(phi(3))*(phi(1)+sqrt(a)*phi(6))*phi(3))...
            + gamma*(conj(phi(3))*phi(3)*(phi(1)+sqrt(a)*phi(6))-conj(phi(2))*phi(2)*(phi(1)+sqrt(a)*phi(6)))...
            -1i*chiM*(2*conj((phi(1)+sqrt(a)*phi(6)))*phi(4)*phi(5)+ conj(phi(4))*(phi(1)+sqrt(a)*phi(6))*phi(4) + conj(phi(5))*(phi(1)+sqrt(a)*phi(6))*phi(5))...
            + gammaM*(conj(phi(5))*phi(5)*(phi(1)+sqrt(a)*phi(6))-conj(phi(4))*phi(4)*(phi(1)+sqrt(a)*phi(6))) ; 
              
  
    dphidt(2) = -1i*omega0*phi(2) -1i*chi*(conj((phi(1)+sqrt(a)*phi(6)))*(phi(1)+sqrt(a)*phi(6))*phi(2)+conj(phi(3))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)))...
            + gamma*(conj(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6))*phi(2)+conj(phi(3))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)));
        



    
    
    dphidt(3) = -1i*omega0*phi(3) -1i*chi*(conj((phi(1)+sqrt(a)*phi(6)))*(phi(1)+sqrt(a)*phi(6))*phi(3)+conj(phi(2))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)))...
                    - gamma*(conj((phi(1)+sqrt(a)*phi(6)))*(phi(1)+sqrt(a)*phi(6))*phi(3)+conj(phi(2))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)));
    
    
    dphidt(4) = -1i*omega0*phi(4) -1i*chiM*(conj((phi(1)+sqrt(a)*phi(6)))*(phi(1)+sqrt(a)*phi(6))*phi(4)+conj(phi(5))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)))...
                    + gammaM*(conj((phi(1)+sqrt(a)*phi(6)))*(phi(1)+sqrt(a)*phi(6))*phi(4)+conj(phi(5))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)));
    dphidt(5) = -1i*omega0*phi(5) -1i*chiM*(conj((phi(1)+sqrt(a)*phi(6)))*(phi(1)+sqrt(a)*phi(6))*phi(5)+conj(phi(4))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)))...
                    - gammaM*(conj((phi(1)+sqrt(a)*phi(6)))*(phi(1)+sqrt(a)*phi(6))*phi(5)+conj(phi(4))*(phi(1)+sqrt(a)*phi(6))*(phi(1)+sqrt(a)*phi(6)));
    
    
    dphidt(6) = -1i*4*omegaRec*phi(6)...
                    -1i*sqrt(a)*chi*(2*conj((phi(1)+sqrt(a)*phi(6)))*phi(2)*phi(3)+ conj(phi(2))*(phi(1)+sqrt(a)*phi(6))*phi(2) + conj(phi(3))*(phi(1)+sqrt(a)*phi(6))*phi(3))...
                    + gamma*sqrt(a)*(conj(phi(3))*phi(3)*(phi(1)+sqrt(a)*phi(6))-conj(phi(2))*phi(2)*(phi(1)+sqrt(a)*phi(6)))...
                    -1i*sqrt(a)*chiM*(2*conj((phi(1)+sqrt(a)*phi(6)))*phi(4)*phi(5)+ conj(phi(4))*(phi(1)+sqrt(a)*phi(6))*phi(4) + conj(phi(5))*(phi(1)+sqrt(a)*phi(6))*phi(5))...
                    + gammaM*sqrt(a)*(conj(phi(5))*phi(5)*(phi(1)+sqrt(a)*phi(6))-conj(phi(4))*phi(4)*(phi(1)+sqrt(a)*phi(6))) ; 
    


end





