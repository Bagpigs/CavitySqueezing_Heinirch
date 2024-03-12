function lambda=rampcoupling(t,tbounds,lambda_1_bounds,ramptype)
tmin=tbounds(1);
tmax=tbounds(2);

% TimeEvolution
%t2=0.015;
t1=0.007;


% vsAtomNr
%t1=0.007;


try
switch ramptype
    case 'rampup_lin'
        lambda= (lambda_1_bounds(2)-lambda_1_bounds(1))* t/(tbounds(2)-tbounds(1)) + lambda_1_bounds(1);
      
    case 'rampup_sshape'
        lambda= (lambda_1_bounds(2)-lambda_1_bounds(1))* (3*(t/(tbounds(2)-tbounds(1))).^2-2*(t/(tbounds(2)-tbounds(1))).^3) + lambda_1_bounds(1);

    case 'rampup_sshape_sqroot'
        lambda= (lambda_1_bounds(2)-lambda_1_bounds(1))* sqrt((3*(t/(tbounds(2)-tbounds(1))).^2-2*(t/(tbounds(2)-tbounds(1))).^3)) + lambda_1_bounds(1);
        
    case 'rampup_sshape_hold'
      tramp=t1;
      if t < tramp   
            lambda= (lambda_1_bounds(2)-lambda_1_bounds(1))* sqrt((3*(t/tramp).^2-2*(t/tramp).^3)) + lambda_1_bounds(1);
            %lambda= (lambda_1_bounds(2)-lambda_1_bounds(1))* sqrt((3*(t/(tbounds(2)-tbounds(1))).^2-2*(t/(tbounds(2)-tbounds(1))).^3)) + lambda_1_bounds(1);
      else
            lambda=lambda_1_bounds(2);
      end      
end
catch error
     fprintf(1,'The identifier was:\n%s',error.identifier);
     fprintf(1,'There was an error! The message was:\n%s',error.message);
     return   
end    
end
%lambda_x and lambda_y
