function la=solve_br(L,mu,nu)
% Solve the bas-relief ambiguity, assuming all lighting have same intensity
L2=[L(1:2,:);-mu*L(1,:)-nu*L(2,:)+L(3,:)].^2;
B=-transpose(L2(1,:)+L2(2,:));
A=[transpose(L2(3,:)),-ones(size(L,2),1)];
x=pinv(A)*B;
la=1/sqrt(x(1));
end
