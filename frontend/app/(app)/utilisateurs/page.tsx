"use client";

import { useState } from "react";
import { CheckCircle2, UserPlus } from "lucide-react";
import { PageHeader } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { SkeletonRows, Table, Td, Th, Tr } from "@/components/ui/data";
import { Input, Label, Select } from "@/components/ui/select";
import { api, getUser } from "@/lib/api";
import { usePolling } from "@/lib/use-polling";
import { cn } from "@/lib/utils";

const ROLES: [string, string][] = [["ANALYSTE", "Analyste"], ["SUPERVISEUR", "Superviseur"], ["ADMIN", "Administrateur"]];
const ROLE_HINT: Record<string, string> = {
  ANALYSTE: "Instruit les alertes et rend les verdicts", SUPERVISEUR: "Réaffecte les dossiers, consulte l'audit",
  ADMIN: "Gère les comptes et les rôles",
};
const initials = (name: string) => name.split(/\s+/).filter(Boolean).slice(0, 2).map((w) => w[0]?.toUpperCase()).join("");

export default function UsersPage() {
  const me = getUser();
  const { data, loading, reload } = usePolling<any[]>("/users", 0);
  const [form, setForm] = useState({ email: "", full_name: "", role: "ANALYSTE", password: "" });
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  async function create(e: React.FormEvent) {
    e.preventDefault(); setMsg(null);
    try {
      await api("/users", { method: "POST", body: JSON.stringify(form) });
      setForm({ ...form, email: "", full_name: "", password: "" }); reload();
      setMsg({ ok: true, text: "Compte créé." });
    } catch (err: any) { setMsg({ ok: false, text: err.message }); }
  }
  async function toggle(u: any) {
    await api(`/users/${u.id}`, { method: "PATCH", body: JSON.stringify({ is_active: !u.is_active }) }); reload();
  }

  return (
    <>
      <PageHeader title="Utilisateurs" subtitle="Comptes du back-office et rôles. Chaque création ou désactivation est inscrite au journal d'audit." />
      <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <Card className="overflow-hidden">
          <Table>
            <thead><tr><Th>Utilisateur</Th><Th>Rôle</Th><Th>État</Th><Th align="right"> </Th></tr></thead>
            <tbody>
              {loading && !data && <SkeletonRows cols={4} rows={4} />}
              {(data ?? []).map((u) => (
                <Tr key={u.id}>
                  <Td>
                    <div className="flex items-center gap-3">
                      <div className={cn("grid h-8 w-8 place-items-center rounded-full text-xs font-semibold", u.is_active ? "bg-primary/12 text-primary" : "bg-surface-2 text-subtle")}>{initials(u.full_name)}</div>
                      <div><div className="text-[13px] font-medium">{u.full_name}{u.id === me?.id && <span className="ml-1.5 text-2xs text-muted-foreground">(vous)</span>}</div>
                        <div className="text-2xs text-muted-foreground">{u.email}</div></div>
                    </div>
                  </Td>
                  <Td><div className="text-[13px]">{ROLES.find(([r]) => r === u.role)?.[1]}</div><div className="text-2xs text-muted-foreground">{ROLE_HINT[u.role]}</div></Td>
                  <Td>
                    <span className={cn("inline-flex items-center gap-1.5 text-xs", u.is_active ? "text-risk-faible" : "text-muted-foreground")}>
                      <span className={cn("h-1.5 w-1.5 rounded-full", u.is_active ? "bg-risk-faible" : "bg-subtle")} />{u.is_active ? "Actif" : "Désactivé"}
                    </span>
                  </Td>
                  <Td align="right">{u.id !== me?.id && <Button size="sm" variant="ghost" onClick={() => toggle(u)}>{u.is_active ? "Désactiver" : "Réactiver"}</Button>}</Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        </Card>
        <Card>
          <CardHeader title="Nouveau compte" description="Le compte est actif dès sa création ; le rôle limite les pages accessibles." actions={<UserPlus className="h-4 w-4 text-subtle" />} />
          <CardContent>
            <form onSubmit={create} className="space-y-3.5">
              <div><Label htmlFor="u-name">Nom complet</Label><Input id="u-name" required value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} /></div>
              <div><Label htmlFor="u-mail">Adresse e-mail</Label><Input id="u-mail" type="email" required value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} /></div>
              <div><Label htmlFor="u-role">Rôle</Label><Select id="u-role" className="w-full" value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })} options={ROLES} />
                <p className="mt-1 text-2xs text-muted-foreground">{ROLE_HINT[form.role]}</p></div>
              <div><Label htmlFor="u-pass">Mot de passe initial</Label><Input id="u-pass" type="password" minLength={10} required placeholder="10 caractères minimum"
                value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></div>
              <Button type="submit" className="w-full"><UserPlus className="h-4 w-4" /> Créer le compte</Button>
              {msg && <p className={cn("flex items-center gap-1.5 text-xs", msg.ok ? "text-risk-faible" : "text-risk-critique")}>{msg.ok && <CheckCircle2 className="h-3.5 w-3.5" />}{msg.text}</p>}
            </form>
          </CardContent>
        </Card>
      </div>
    </>
  );
}
